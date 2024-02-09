import torch
import torch.nn as nn
from torch import Tensor
from layers import MultiHeadAttention, MultiHeadAttentionStatic

def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation(), nn.Dropout(p=0.3)])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
        del net_list[-1]
    return nn.Sequential(*net_list)

def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)

class ActorMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        # MLP
        self.state_dim = args.n_stocks * args.state_dim
        self.action_dim = args.n_stocks
        self.dims = list(map(int, args.net_dims.split(',')))
        self.net = build_mlp(dims=[self.state_dim, *self.dims, self.action_dim])
        # layer_init_with_orthogonal(self.net[-1], std=0.1)
    
    def forward(self, state: Tensor):
        batch_size, n_stocks, window, n_features = state.shape
        state = state[:, :, -1, :].reshape(batch_size, -1)
        action_avg = self.net(state).squeeze(-1)
        return action_avg

class ActorMLPv2(ActorMLP):
    def __init__(self, args):
        super().__init__(args)
    
    def forward(self, state: Tensor):
        batch_size, n_stocks, window, n_features = state.shape
        state = state.transpose(1, 2).reshape(batch_size, window, -1)
        action_avg = self.net(state)
        return action_avg

class LSTMHA(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.d_model, self.d_k = list(map(int,args.net_dims.split(',')))
        self.d_v = self.d_k
        self.seq_nn = nn.LSTM(args.state_dim, self.d_model, batch_first=True)
        self.attn = MultiHeadAttention(args.n_head, self.d_model, self.d_k, self.d_v, 0.0)
        # self.out_layer = nn.Linear(self.d_model, 1)
        self.out_layer = nn.Sequential()
        self.out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.out_layer.append(nn.ReLU())
        self.out_layer.append(nn.Linear(self.d_model, 1))
        self.return_out_layer = nn.Sequential()
        self.return_out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.return_out_layer.append(nn.ReLU())
        self.return_out_layer.append(nn.Linear(self.d_model, 1))
    
    def forward(self, state: Tensor, assets_cov: Tensor):
        batch_size, n_stocks, window, n_features = state.shape
        reshaped_state = state.reshape(-1, window, n_features)
        _, (hn, _) = self.seq_nn(reshaped_state)
        hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn)
        # hn: [batch_size, n_stocks]
        r = self.return_out_layer(hn).squeeze(-1)
        hn = self.out_layer(hn).squeeze(-1)
        return hn, r
    
class LSTMHAv2(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_stocks = args.n_stocks
        self.d_model, self.d_k = list(map(int,args.net_dims.split(',')))
        self.d_v = self.d_k
        self.seq_nn = nn.LSTM(args.state_dim, self.d_model, batch_first=True)
        self.attn = MultiHeadAttention(args.n_head, self.d_model, self.d_k, self.d_v, 0.0)
        # self.out_layer = nn.Linear(self.d_model + self.n_stocks, 1)
        self.out_layer = nn.Linear(self.d_model, 1)
    
    def forward(self, state: Tensor):
        # state: [batch_size, n_stocks, window, n_features]
        batch_size, n_stocks, window, n_features = state.shape
        # reshaped_state: [batch_size * n_stocks, window, n_features]
        reshaped_state = state.reshape(-1, window, n_features)
        _, (hn, _) = self.seq_nn(reshaped_state)
        # hn: [batch_size, n_stocks, d_model]
        hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn)
        # hn: [batch_size, n_stocks]
        # action = torch.zeros(self.n_stocks, dtype=torch.float, device=state.device)
        # actions = []
        # for i in range(hn.shape[0]):
        #     # fusion_hn: [n_stocks, d_model + n_stocks]
        #     fusion_hn = torch.cat([hn[i, :, :], action.unsqueeze(0).repeat(self.n_stocks, 1).detach()], dim=-1)
        #     # action: [n_stocks, 1]
        #     action = self.out_layer(fusion_hn).squeeze(-1)
        #     actions.append(action)
        # # actions: [batch_size, n_stocks]
        # actions = torch.stack(actions, dim=0)
        # hn = self.out_layer(hn).squeeze(-1)
        actions = self.out_layer(hn).squeeze(-1)
        return actions

class LSTMHAv3(LSTMHA):
    def __init__(self, args):
        super().__init__(args)

class HA(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.n_stocks = args.n_stocks
        self.d_model, self.d_k = list(map(int,args.net_dims.split(',')))
        self.d_v = self.d_k
        # self.seq_nn = nn.LSTM(args.state_dim, self.d_model, batch_first=True)
        self.feat_enc = nn.Linear(args.state_dim, self.d_model)
        self.attn = MultiHeadAttention(args.n_head, self.d_model, self.d_k, self.d_v, 0.0)
        # self.out_layer = nn.Linear(self.d_model + self.n_stocks, 1)
        self.out_layer = nn.Linear(self.d_model, 1)
        # self.dynamic_matrix = nn.Linear(self.n_stocks, self.n_stocks, bias=False)
        # self.stock_embeddings = nn.Embedding(self.n_stocks, self.d_model)
    
    def forward(self, state: Tensor):
        # state: [batch_size, n_stocks, window, n_features]
        batch_size, n_stocks, window, n_features = state.shape
        # reshaped_state: [batch_size * n_stocks, window, n_features]
        # reshaped_state = state.reshape(-1, window, n_features)
        # _, (hn, _) = self.seq_nn(reshaped_state)
        # hn: [batch_size, n_stocks, d_model]
        # hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, n_features]
        hn = state[:, :, -1, :]
        # hn: [batch_size, n_stocks, d_model]
        hn = self.feat_enc(hn)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn)
        # hn: [batch_size, n_stocks]
        # action = torch.zeros(self.n_stocks, dtype=torch.float, device=state.device)
        # actions = []
        # for i in range(hn.shape[0]):
        #     # fusion_hn: [n_stocks, d_model + n_stocks]
        #     fusion_hn = torch.cat([hn[i, :, :], action.unsqueeze(0).repeat(self.n_stocks, 1).detach()], dim=-1)
        #     # action: [n_stocks, 1]
        #     action = self.out_layer(fusion_hn).squeeze(-1)
        #     actions.append(action)
        # # actions: [batch_size, n_stocks]
        # actions = torch.stack(actions, dim=0)
        # hn = self.out_layer(hn).squeeze(-1)
        # cov_matrix: [n_stocks, n_stocks]
        # cov_matrix = (self.stock_embeddings.weight @ self.stock_embeddings.weight.T).softmax(-1)
        # hn = (hn.transpose(1, 2) @ cov_matrix).transpose(1, 2)
        actions = self.out_layer(hn).squeeze(-1)
        return actions

class LSTMHADW(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.d_model, self.d_k = list(map(int,args.net_dims.split(',')))
        self.d_v = self.d_k
        self.seq_nn = nn.LSTM(args.state_dim, self.d_model, batch_first=True)
        self.attn = MultiHeadAttention(args.n_head, self.d_model, self.d_k, self.d_v, 0.0)
        # self.out_layer = nn.Linear(self.d_model, 1)
        self.out_layer = nn.Sequential()
        self.out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.out_layer.append(nn.ReLU())
        # self.out_layer.append(nn.Tanh())
        self.out_layer.append(nn.Linear(self.d_model, 1))
        self.return_out_layer = nn.Sequential()
        self.return_out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.return_out_layer.append(nn.ReLU())
        # self.return_out_layer.append(nn.Tanh())
        self.return_out_layer.append(nn.Linear(self.d_model, 1))
        # self.mov_out_layer = nn.Sequential()
        # self.mov_out_layer.append(nn.Linear(self.d_model, self.d_model))
        # self.mov_out_layer.append(nn.ReLU())
        # self.mov_out_layer.append(nn.Linear(self.d_model, 1))
        # We only have three tasks
        self.dw = nn.Parameter(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float))
    
    def forward(self, state: Tensor, assets_cov: Tensor):
        batch_size, n_stocks, window, n_features = state.shape
        reshaped_state = state.reshape(-1, window, n_features)
        _, (hn, _) = self.seq_nn(reshaped_state)
        hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn)
        # hn: [batch_size, n_stocks]
        r = self.return_out_layer(hn).squeeze(-1)
        # m: [batch_size, n_stocks] for movement prediction
        # mp = self.mov_out_layer(hn).squeeze(-1)
        hn = self.out_layer(hn).squeeze(-1)
        # assets_cov: [batch_size, n_stocks, n_stocks] @ hn: [batch_size, n_stocks]
        # hn = hn - ((assets_cov + assets_cov.transpose(1, 2)) @ hn.unsqueeze(-1)).squeeze(-1)
        return hn, r #, mp
    
    def forward_with_hiddenstate(self, state, assets_cov):
        batch_size, n_stocks, window, n_features = state.shape
        reshaped_state = state.reshape(-1, window, n_features)
        _, (hn, _) = self.seq_nn(reshaped_state)
        hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn)
        # hn: [batch_size, n_stocks]
        r = self.return_out_layer(hn).squeeze(-1)
        # m: [batch_size, n_stocks] for movement prediction
        # mp = self.mov_out_layer(hn).squeeze(-1)
        hn_action = self.out_layer(hn).squeeze(-1)
        return hn_action, r, hn

class LSTMHADWQuantile(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.d_model, self.d_k = list(map(int,args.net_dims.split(',')))
        self.d_v = self.d_k
        self.seq_nn = nn.LSTM(args.state_dim, self.d_model, batch_first=True)
        self.attn = MultiHeadAttention(args.n_head, self.d_model, self.d_k, self.d_v, 0.0)
        # self.out_layer = nn.Linear(self.d_model, 1)
        self.out_layer = nn.Sequential()
        self.out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.out_layer.append(nn.ReLU())
        # self.out_layer.append(nn.Tanh())
        self.out_layer.append(nn.Linear(self.d_model, 1))
        self.return_out_layer = nn.Sequential()
        self.return_out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.return_out_layer.append(nn.ReLU())
        # self.return_out_layer.append(nn.Tanh())
        self.return_out_layer.append(nn.Linear(self.d_model, args.n_quantile))
        # self.mov_out_layer = nn.Sequential()
        # self.mov_out_layer.append(nn.Linear(self.d_model, self.d_model))
        # self.mov_out_layer.append(nn.ReLU())
        # self.mov_out_layer.append(nn.Linear(self.d_model, 1))
        # We only have three tasks
        self.dw = nn.Parameter(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float))
    
    def forward(self, state: Tensor, assets_cov: Tensor):
        batch_size, n_stocks, window, n_features = state.shape
        reshaped_state = state.reshape(-1, window, n_features)
        _, (hn, _) = self.seq_nn(reshaped_state)
        hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn)
        # hn: [batch_size, n_stocks]
        r = self.return_out_layer(hn).squeeze(-1)
        hn = self.out_layer(hn).squeeze(-1)
        # assets_cov: [batch_size, n_stocks, n_stocks] @ hn: [batch_size, n_stocks]
        # hn = hn - ((assets_cov + assets_cov.transpose(1, 2)) @ hn.unsqueeze(-1)).squeeze(-1)
        return hn, r

class LSTMHASCDW(nn.Module):
    # Static cov in Attention
    def __init__(self, args):
        super().__init__()

        self.d_model, self.d_k = list(map(int,args.net_dims.split(',')))
        self.d_v = self.d_k
        self.seq_nn = nn.LSTM(args.state_dim, self.d_model, batch_first=True)
        self.attn = MultiHeadAttentionStatic(args.n_head, self.d_model, self.d_k, self.d_v, 0.0)
        # self.out_layer = nn.Linear(self.d_model, 1)
        self.out_layer = nn.Sequential()
        self.out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.out_layer.append(nn.ReLU())
        # self.out_layer.append(nn.Tanh())
        self.out_layer.append(nn.Linear(self.d_model, 1))
        self.return_out_layer = nn.Sequential()
        self.return_out_layer.append(nn.Linear(self.d_model, self.d_model))
        self.return_out_layer.append(nn.ReLU())
        # self.return_out_layer.append(nn.Tanh())
        self.return_out_layer.append(nn.Linear(self.d_model, 1))
        # self.mov_out_layer = nn.Sequential()
        # self.mov_out_layer.append(nn.Linear(self.d_model, self.d_model))
        # self.mov_out_layer.append(nn.ReLU())
        # self.mov_out_layer.append(nn.Linear(self.d_model, 1))
        # We only have three tasks
        self.dw = nn.Parameter(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float))
    
    def forward(self, state: Tensor, assets_cov: Tensor):
        batch_size, n_stocks, window, n_features = state.shape
        reshaped_state = state.reshape(-1, window, n_features)
        _, (hn, _) = self.seq_nn(reshaped_state)
        hn = hn.squeeze(0).reshape(batch_size, n_stocks, -1)
        # hn: [batch_size, n_stocks, d_model]
        hn, _ = self.attn(hn, hn, hn, assets_cov)
        # hn: [batch_size, n_stocks]
        r = self.return_out_layer(hn).squeeze(-1)
        # m: [batch_size, n_stocks] for movement prediction
        # mp = self.mov_out_layer(hn).squeeze(-1)
        hn = self.out_layer(hn).squeeze(-1)
        # assets_cov: [batch_size, n_stocks, n_stocks] @ hn: [batch_size, n_stocks]
        # hn = hn - ((assets_cov + assets_cov.transpose(1, 2)) @ hn.unsqueeze(-1)).squeeze(-1)
        return hn, r #, mp
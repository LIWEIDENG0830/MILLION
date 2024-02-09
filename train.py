import torch
import copy
import model as model_factory
from test import eval_model, eval_model_with_costv2, eval_model_with_action_optimize
from tensorboardX import SummaryWriter
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
from utils import load_torch_file

# 1. Compare reinforcement learning with the proposed framework under the same model and reward -> Double E
# 2. Performance of different models under the proposed framework   -> Flexible
# 3. Performance of different objectives                            -> Flexible
# 4. Multi-Objective (Optional)
# 5. Hard Constraint (Optional)
# 6. Multi-Market (Optional)

# Code Check
# Ranking Loss
# Multiple Datasets

# TODO:
# Ranking Loss + Prediction Loss + Classification Loss

def train_one_epoch_with_action_constraint(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _, _ = train_loader
    transaction_cost_rate = 0.0
    if args.trans_cost:
        transaction_cost_rate = args.trans_rate
    for batch in train_loader.get_batches():
        state, diff, _, assets_cov = [item.to(device) for item in batch]
        state = state.squeeze(0)
        diff = diff.squeeze(0)
        # action_t-1 + state_t -> action_t
        # state_t -> action_t  == > action_t-1
        # action_t-1
        # action: [batch_size, n_stocks]
        action, y_r = model(state, assets_cov)
        action = action.softmax(-1)
        batch_size, n_stocks = action.shape
        # simulate the investment in window size days
        # total_assets = torch.ones((batch_size,), device=action.device)
        total_assets = 1
        # total_assets_list = []
        for i in range(action.shape[0]):
            # transaction cost
            if i == 0:
                # window size == total length of training data
                # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
                # total_assets = (return_rate + 1) * total_assets
                return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
                total_assets = (return_rate + 1) * total_assets
                # reward = (action[:, i, :] * total_assets * diff).sum(-1)
            else:
                # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
                # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
                return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
                # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
                total_assets = (return_rate + 1) * total_assets
            # total_assets_list.append(total_assets)
        # total_assets_list = torch.stack(total_assets_list)
        # return rate
        # period_return_rate = total_assets_list[1:] - total_assets_list[:-1]
        # Assume the risk free rate is zero
        # sharpe + reward -> HeatMap
        # Framework -> Models -> Models Performance
        # 3. Model Design (Optional)
        # 2. Different Reward
        # 1. Pretraining + Directly Ranking -> Multiple Objectives
        # RL Weakness: Hard to generalize to multiple objective
        # Downside Risk
        # reward = period_return_rate.mean() / period_return_rate.std()
        # reward = total_assets - 1
        # loss = torch.mean(-reward)
        mse_loss = F.mse_loss(y_r, diff-1)
        ranking_loss = F.relu(-(y_r.unsqueeze(-1) - y_r.unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
        ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
        penalty = F.relu(action - args.action_constraint).sum()
        reward = total_assets - penalty * args.action_constraint_weight
        # DW refers dynamic weight
        if args.model_name.endswith('DW'):
            loss_weight = 1 / (model.dw * model.dw)
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[1:]).sum()
            loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
        else:
            loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        # train_reward += period_return_rate.mean().item() * action.shape[0]
        train_reward += total_assets.item()
        n_samples += action.shape[0]
    return train_reward / n_samples

def train_one_epoch_with_downside_risk(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _, _ = train_loader
    transaction_cost_rate = 0.0
    if args.trans_cost:
        transaction_cost_rate = args.trans_rate
    for batch in train_loader.get_batches():
        state, diff, _, assets_cov = [item.to(device) for item in batch]
        state = state.squeeze(0)
        diff = diff.squeeze(0)
        # action_t-1 + state_t -> action_t
        # state_t -> action_t  == > action_t-1
        # action_t-1
        # action: [batch_size, n_stocks]
        # action, y_r, mp = model(state, assets_cov)
        action, y_r = model(state, assets_cov)
        action = action.softmax(-1)
        batch_size, n_stocks = action.shape
        # simulate the investment in window size days
        # total_assets = torch.ones((batch_size,), device=action.device)
        total_assets = 1
        total_assets_list = []
        for i in range(action.shape[0]):
            # transaction cost
            if i == 0:
                # window size == total length of training data
                # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
                # total_assets = (return_rate + 1) * total_assets
                return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
                total_assets = (return_rate + 1) * total_assets
                # reward = (action[:, i, :] * total_assets * diff).sum(-1)
            else:
                # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
                # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
                return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
                # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
                total_assets = (return_rate + 1) * total_assets
            total_assets_list.append(total_assets)
        total_assets_list = torch.stack(total_assets_list)
        # return rate
        period_return_rate = (total_assets_list[1:] - total_assets_list[:-1]) / total_assets_list[:-1]

        # cov risk -> Portfolio risk
        # state -> Risk-aware action -> 
        # risk -> portfolio std 
        # state -> action dist -> actions -> a set of portfolio return -> portfolio std

        # a sequence of state -> a set of portflio return -> action_t -> return rate -> portfolio std
        # action_t -> risk <- portfolio std
        # action_t -> y_t -> reward

        # diversity
        # cov_constraint = (action.unsqueeze(1) @ assets_cov @ action.unsqueeze(-1)).sum()
        # Assume the risk free rate is zero
        # sharpe + reward -> HeatMap
        # Framework -> Models -> Models Performance
        # 3. Model Design (Optional)
        # 2. Different Reward
        # 1. Pretraining + Directly Ranking -> Multiple Objectives
        # RL Weakness: Hard to generalize to multiple objective
        # Downside Risk
        # reward = period_return_rate.mean() / period_return_rate.std()
        # mse_loss_p = F.mse_loss((action * y_r).sum(-1), (action * (diff-1)).sum(-1).detach())
        # mp_loss = F.binary_cross_entropy_with_logits(mp, ((diff-1)>0).float())
        mse_loss = F.mse_loss(y_r, diff-1)
        # mse_loss = F.l1_loss(y_r, diff-1)
        ranking_loss = F.relu(-(y_r.unsqueeze(-1) - y_r.unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
        ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
        # where the alpha_downside can be dynamically set, e.g., setting as a benchmark return.
        # reward = - F.relu(- (period_return_rate - args.alpha_downside)).sum() - args.cov_weight * cov_constraint
        reward = - F.relu(- (period_return_rate - args.alpha_downside)).sum()
        # reward = total_assets - 1
        # loss = torch.mean(-reward)
        # DW refers dynamic weight
        if args.model_name.endswith('DW'):
            loss_weight = 1 / (3 * model.dw * model.dw)
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[2:]).sum()
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw[2:]).sum()
            # loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + loss_weight[3] * mp_loss + torch.log(model.dw).sum()
            loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
        else:
            loss = - reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # loss = reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        train_reward += period_return_rate.mean().item() * action.shape[0]
        n_samples += action.shape[0]
    return train_reward / n_samples

def train_one_epoch_with_downside_risk_quantile(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _, _ = train_loader
    transaction_cost_rate = 0.0
    if args.trans_cost:
        transaction_cost_rate = args.trans_rate
    for batch in train_loader.get_batches():
        state, diff, _, assets_cov = [item.to(device) for item in batch]
        state = state.squeeze(0)
        diff = diff.squeeze(0)
        # action_t-1 + state_t -> action_t
        # state_t -> action_t  == > action_t-1
        # action_t-1
        # action: [batch_size, n_stocks]
        # y_r: [batch_size, n_stocks, n_quantile]
        action, y_r = model(state, assets_cov)
        action = action.softmax(-1)
        batch_size, n_stocks = action.shape
        # simulate the investment in window size days
        # total_assets = torch.ones((batch_size,), device=action.device)
        total_assets = 1
        total_assets_list = []
        for i in range(action.shape[0]):
            # transaction cost
            if i == 0:
                # window size == total length of training data
                # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
                # total_assets = (return_rate + 1) * total_assets
                return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
                total_assets = (return_rate + 1) * total_assets
                # reward = (action[:, i, :] * total_assets * diff).sum(-1)
            else:
                # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
                # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
                return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
                # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
                total_assets = (return_rate + 1) * total_assets
            total_assets_list.append(total_assets)
        total_assets_list = torch.stack(total_assets_list)
        # return rate
        period_return_rate = (total_assets_list[1:] - total_assets_list[:-1]) / total_assets_list[:-1]

        # cov risk -> Portfolio risk
        # state -> Risk-aware action -> 
        # risk -> portfolio std 
        # state -> action dist -> actions -> a set of portfolio return -> portfolio std

        # a sequence of state -> a set of portflio return -> action_t -> return rate -> portfolio std
        # action_t -> risk <- portfolio std
        # action_t -> y_t -> reward

        # diversity
        # cov_constraint = (action.unsqueeze(1) @ assets_cov @ action.unsqueeze(-1)).sum()
        # Assume the risk free rate is zero
        # sharpe + reward -> HeatMap
        # Framework -> Models -> Models Performance
        # 3. Model Design (Optional)
        # 2. Different Reward
        # 1. Pretraining + Directly Ranking -> Multiple Objectives
        # RL Weakness: Hard to generalize to multiple objective
        # Downside Risk
        # reward = period_return_rate.mean() / period_return_rate.std()
        # mse_loss_p = F.mse_loss((action * y_r).sum(-1), (action * (diff-1)).sum(-1).detach())
        # mse_loss = F.mse_loss(y_r, diff-1)
        # quantile loss of y_r
        mse_loss_p5 = (0.05 * F.relu(y_r[:, :, 0] - diff + 1) + 0.95 * F.relu(diff-1 - y_r[:, :, 0])).mean()
        mse_loss_p50 = F.l1_loss(y_r[:, :, 1], diff-1, reduction='none').mean()
        mse_loss_p95 = (0.95 * F.relu(y_r[:, :, 2] - diff + 1) + 0.05 * F.relu(diff-1 - y_r[:, :, 2])).mean()
        mse_loss = mse_loss_p50 + mse_loss_p5 + mse_loss_p95
        # mse_loss = F.l1_loss(y_r, diff-1)
        ranking_loss = F.relu(-(y_r[:, :, 1].unsqueeze(-1) - y_r[:, :, 1].unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
        ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
        # where the alpha_downside can be dynamically set, e.g., setting as a benchmark return.
        # reward = - F.relu(- (period_return_rate - args.alpha_downside)).sum() - args.cov_weight * cov_constraint
        reward = - F.relu(- (period_return_rate - args.alpha_downside)).sum()
        # reward = total_assets - 1
        # loss = torch.mean(-reward)
        # DW refers dynamic weight
        if args.model_name.endswith('DW'):
            loss_weight = 1 / (3 * model.dw * model.dw)
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[2:]).sum()
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw[2:]).sum()
            loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
        else:
            loss_weight = 1 / (3 * model.dw * model.dw)
            loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
            # loss = - reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # loss = reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        train_reward += period_return_rate.mean().item() * action.shape[0]
        n_samples += action.shape[0]
    return train_reward / n_samples

# def train_one_epoch_with_downside_risk_dy(args, train_loader, model, optimizer, device):
#     # dynamic downside risk threshold
#     model.train()
#     train_reward = 0
#     n_samples = 0
#     train_loader, _, _, _, _ = train_loader
#     transaction_cost_rate = 0.0
#     if args.trans_cost:
#         transaction_cost_rate = args.trans_rate
#     for batch in train_loader.get_batches():
#         state, diff, _, assets_cov = [item.to(device) for item in batch]
#         state = state.squeeze(0)
#         diff = diff.squeeze(0)
#         # action_t-1 + state_t -> action_t
#         # state_t -> action_t  == > action_t-1
#         # action_t-1
#         # action: [batch_size, n_stocks]
#         action, y_r = model(state, assets_cov)
#         action = action.softmax(-1)
#         batch_size, n_stocks = action.shape
#         # simulate the investment in window size days
#         # total_assets = torch.ones((batch_size,), device=action.device)
#         total_assets = 1
#         total_assets_list = [torch.tensor(1, dtype=torch.float).detach().to(device)]
#         for i in range(action.shape[0]):
#             # transaction cost
#             if i == 0:
#                 # window size == total length of training data
#                 # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
#                 # total_assets = (return_rate + 1) * total_assets
#                 return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
#                 total_assets = (return_rate + 1) * total_assets
#                 # reward = (action[:, i, :] * total_assets * diff).sum(-1)
#             else:
#                 # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
#                 # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
#                 return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
#                 # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
#                 total_assets = (return_rate + 1) * total_assets
#             total_assets_list.append(total_assets)
#         total_assets_list = torch.stack(total_assets_list)
#         # return rate
#         period_return_rate = (total_assets_list[1:] - total_assets_list[:-1]) / total_assets_list[:-1]

#         # cov risk -> Portfolio risk
#         # state -> Risk-aware action -> 
#         # risk -> portfolio std 
#         # state -> action dist -> actions -> a set of portfolio return -> portfolio std

#         # a sequence of state -> a set of portflio return -> action_t -> return rate -> portfolio std
#         # action_t -> risk <- portfolio std
#         # action_t -> y_t -> reward

#         # diversity
#         # cov_constraint = (action.unsqueeze(1) @ assets_cov @ action.unsqueeze(-1)).sum()
#         # Assume the risk free rate is zero
#         # sharpe + reward -> HeatMap
#         # Framework -> Models -> Models Performance
#         # 3. Model Design (Optional)
#         # 2. Different Reward
#         # 1. Pretraining + Directly Ranking -> Multiple Objectives
#         # RL Weakness: Hard to generalize to multiple objective
#         # Downside Risk
#         # reward = period_return_rate.mean() / period_return_rate.std()
#         # mse_loss_p = F.mse_loss((action * y_r).sum(-1), (action * (diff-1)).sum(-1).detach())
#         mse_loss = F.mse_loss(y_r, diff-1)
#         # mse_loss = F.l1_loss(y_r, diff-1)
#         ranking_loss = F.relu(-(y_r.unsqueeze(-1) - y_r.unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
#         ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
#         # where the alpha_downside can be dynamically set, e.g., setting as a benchmark return.
#         # reward = - F.relu(- (period_return_rate - args.alpha_downside)).sum() - args.cov_weight * cov_constraint
#         threshold = F.relu(torch.quantile(diff-1, q=0.8, dim=-1))
#         reward = - F.relu(- (period_return_rate - threshold)).sum()
#         # reward = total_assets - 1
#         # loss = torch.mean(-reward)
#         # DW refers dynamic weight
#         if args.model_name.endswith('DW'):
#             loss_weight = 1 / (3 * model.dw * model.dw)
#             # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[2:]).sum()
#             # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw[2:]).sum()
#             loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
#         else:
#             loss = - reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
#         # loss = reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Record Training Information
#         train_reward += period_return_rate.mean().item() * action.shape[0]
#         n_samples += action.shape[0]
#     return train_reward / n_samples

def train_one_epoch_with_sharpe(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _, _ = train_loader
    # transaction_cost_rate = 0.001
    transaction_cost_rate = 0.0
    if args.trans_cost:
        transaction_cost_rate = args.trans_rate
    for batch in train_loader.get_batches():
        state, diff, _, assets_cov = [item.to(device) for item in batch]
        state = state.squeeze(0)
        diff = diff.squeeze(0)
        # action_t-1 + state_t -> action_t
        # state_t -> action_t  == > action_t-1
        # action_t-1
        # action: [batch_size, n_stocks]
        action, y_r = model(state, assets_cov)
        action = action.softmax(-1)
        batch_size, n_stocks = action.shape
        # simulate the investment in window size days
        # total_assets = torch.ones((batch_size,), device=action.device)
        total_assets = 1
        total_assets_list = []
        for i in range(action.shape[0]):
            # transaction cost
            if i == 0:
                # window size == total length of training data
                # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
                # total_assets = (return_rate + 1) * total_assets
                return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
                total_assets = (return_rate + 1) * total_assets
                # reward = (action[:, i, :] * total_assets * diff).sum(-1)
            else:
                # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
                # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
                return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
                # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
                total_assets = (return_rate + 1) * total_assets
            total_assets_list.append(total_assets)
        total_assets_list = torch.stack(total_assets_list)
        # return rate
        # Settings: Personalized Risk Controllable Portfolio Management with Adaptive Constraint Learning
        # Domain Adaptive
        # A -> f(A) safe-layer -> sigma <- A sigma_matrix
        # A -> f(A) safe-layer -> sigma <- sigma'
        # sigma - sigma' gap -> f(A) safe layer -> A'
        # Action Correction
        # A -> f'(A) reward-layer -> reward
        period_return_rate = (total_assets_list[1:] - total_assets_list[:-1]) / total_assets_list[:-1]
        # Assume the risk free rate is zero
        # sharpe + reward -> HeatMap
        # Framework -> Models -> Models Performance
        # 3. Model Design (Optional)
        # 2. Different Reward
        # 1. Pretraining + Directly Ranking -> Multiple Objectives
        # RL Weakness: Hard to generalize to multiple objective
        # Downside Risk
        mse_loss = F.mse_loss(y_r, diff-1)
        ranking_loss = F.relu(-(y_r.unsqueeze(-1) - y_r.unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
        ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
        reward = period_return_rate.mean() / period_return_rate.std()
        # reward = total_assets - 1
        # loss = torch.mean(-reward)
        # loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # DW refers dynamic weight
        if args.model_name.endswith('DW'):
            loss_weight = 1 / (model.dw * model.dw)
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[1:]).sum()
            loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw[1:3]).sum()
        else:
            loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        train_reward += period_return_rate.mean().item() * action.shape[0]
        n_samples += action.shape[0]
    return train_reward / n_samples

def train_one_epoch_with_cumulative_return(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _, _ = train_loader
    transaction_cost_rate = 0.0
    if args.trans_cost:
        transaction_cost_rate = args.trans_rate
    for batch in train_loader.get_batches():
        state, diff, _, assets_cov = [item.to(device) for item in batch]
        state = state.squeeze(0)
        diff = diff.squeeze(0)
        # action_t-1 + state_t -> action_t
        # state_t -> action_t  == > action_t-1
        # action_t-1
        # action: [batch_size, n_stocks]
        action, y_r = model(state, assets_cov)
        action = action.softmax(-1)
        # batch_size, n_stocks = action.shape
        # simulate the investment in window size days
        # total_assets = torch.ones((batch_size,), device=action.device)
        total_assets = 1
        # total_assets_list = []
        for i in range(action.shape[0]):
            # transaction cost
            if i == 0:
                # window size == total length of training data
                # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
                # total_assets = (return_rate + 1) * total_assets
                return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
                total_assets = (return_rate + 1) * total_assets
                # reward = (action[:, i, :] * total_assets * diff).sum(-1)
            else:
                # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
                # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
                return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
                # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
                total_assets = (return_rate + 1) * total_assets
            # total_assets_list.append(total_assets)

        # different action present different return
        # cov_constraint = (action.unsqueeze(1) @ assets_cov @ action.unsqueeze(-1)).sum()

        # total_assets_list = torch.stack(total_assets_list)
        # return rate
        # period_return_rate = total_assets_list[1:] - total_assets_list[:-1]
        # Assume the risk free rate is zero
        # sharpe + reward -> HeatMap
        # Framework -> Models -> Models Performance
        # 3. Model Design (Optional)
        # 2. Different Reward
        # 1. Pretraining + Directly Ranking -> Multiple Objectives
        # RL Weakness: Hard to generalize to multiple objective
        # Downside Risk
        # reward = period_return_rate.mean() / period_return_rate.std()
        # reward = total_assets - 1
        # loss = torch.mean(-reward)
        mse_loss = F.mse_loss(y_r, diff-1)
        ranking_loss = F.relu(-(y_r.unsqueeze(-1) - y_r.unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
        ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
        # reward = total_assets - args.cov_weight * cov_constraint
        reward = total_assets
        # DW refers dynamic weight
        if args.model_name.endswith('DW'):
            loss_weight = 1 / (3 * model.dw * model.dw)
            # loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[1:]).sum()
            loss = - loss_weight[0] * reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(model.dw).sum()
        else:
            loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # reward = total_assets
        # loss = -reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        # train_reward += period_return_rate.mean().item() * action.shape[0]
        train_reward += reward.item()
        n_samples += action.shape[0]
    return train_reward / n_samples

def train_one_epoch_with_cumulative_return_short(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _ = train_loader
    transaction_cost_rate = 0.0
    if args.trans_cost:
        transaction_cost_rate = args.trans_rate
    for batch in train_loader.get_batches():
        state, diff, _ = [item.to(device) for item in batch]
        state = state.squeeze(0)
        diff = diff.squeeze(0)
        # action_t-1 + state_t -> action_t
        # state_t -> action_t  == > action_t-1
        # action_t-1
        # action: [batch_size, n_stocks]
        action, y_r = model(state)
        # action = action / torch.sum((F.relu(action) + F.relu(-action)), dim=-1, keepdim=True)
        action = torch.sign(action).detach() * action.softmax(-1)
        # batch_size, n_stocks = action.shape
        # simulate the investment in window size days
        # total_assets = torch.ones((batch_size,), device=action.device)
        total_assets = 1
        # total_assets_list = []
        for i in range(action.shape[0]):
            # transaction cost
            if i == 0:
                # window size == total length of training data
                # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
                # total_assets = (return_rate + 1) * total_assets
                return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
                total_assets = (return_rate + 1) * total_assets
                # reward = (action[:, i, :] * total_assets * diff).sum(-1)
            else:
                # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
                # return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
                return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
                # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
                total_assets = (return_rate + 1) * total_assets
            # total_assets_list.append(total_assets)
        # total_assets_list = torch.stack(total_assets_list)
        # return rate
        # period_return_rate = total_assets_list[1:] - total_assets_list[:-1]
        # Assume the risk free rate is zero
        # sharpe + reward -> HeatMap
        # Framework -> Models -> Models Performance
        # 3. Model Design (Optional)
        # 2. Different Reward
        # 1. Pretraining + Directly Ranking -> Multiple Objectives
        # RL Weakness: Hard to generalize to multiple objective
        # Downside Risk
        # reward = period_return_rate.mean() / period_return_rate.std()
        # reward = total_assets - 1
        # loss = torch.mean(-reward)

        mse_loss = F.mse_loss(y_r, diff-1)
        ranking_loss = F.relu(-(y_r.unsqueeze(-1) - y_r.unsqueeze(1)) * (diff.unsqueeze(-1) - diff.unsqueeze(1)))
        ranking_loss = ranking_loss.sum(-1).sum(-1).mean()
        reward = total_assets
        # DW refers dynamic weight
        if args.model_name.endswith('DW'):
            loss_weight = 1 / (model.dw * model.dw)
            loss = - reward + loss_weight[1] * mse_loss + loss_weight[2] * ranking_loss + torch.log(loss_weight[1:]).sum()
        else:
            loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # loss = -reward + args.mse_alpha * mse_loss + args.ranking_alpha * ranking_loss
        # reward = total_assets
        # loss = -reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        # train_reward += period_return_rate.mean().item() * action.shape[0]
        train_reward += reward.item()
        n_samples += action.shape[0]
    return train_reward / n_samples

# def train_one_epoch_with_one_step_reward(args, train_loader, model, optimizer, device):
#     model.train()
#     train_reward = 0
#     n_samples = 0
#     train_loader, _, _, _ = train_loader
#     transaction_cost_rate = 0.0
#     if args.trans_cost:
#         transaction_cost_rate = args.trans_rate
#     for batch in train_loader.get_batches():
#         state, diff, _ = [item.to(device) for item in batch]
#         state = state.squeeze(0)
#         diff = diff.squeeze(0)
#         # action_t-1 + state_t -> action_t
#         # state_t -> action_t  == > action_t-1
#         # action_t-1
#         # action: [batch_size, n_stocks]
#         action = model(state).softmax(-1)
#         # return_rate: [batch_size]
#         return_rate = (action * (diff - 1)).sum() + 1
#         # batch_size, n_stocks = action.shape
#         # simulate the investment in window size days
#         # total_assets = torch.ones((batch_size,), device=action.device)
#         # total_assets = 1
#         # total_assets_list = []
#         # for i in range(action.shape[0]):
#             # transaction cost
#             # if i == 0:
#             #     # window size == total length of training data
#             #     # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
#             #     # total_assets = (return_rate + 1) * total_assets
#             #     return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
#             #     total_assets = (return_rate + 1) * total_assets
#             #     # reward = (action[:, i, :] * total_assets * diff).sum(-1)
#             # else:
#             #     # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
#             #     return_rate = ((diff[i, :] - 1) * action[i, :] - (action[i, :] - action[i-1, :]) * transaction_cost_rate).sum(-1)
#             #     # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
#             #     total_assets = (return_rate + 1) * total_assets
#             # total_assets_list.append(total_assets)
#         # total_assets_list = torch.stack(total_assets_list)
#         # return rate
#         # period_return_rate = total_assets_list[1:] - total_assets_list[:-1]
#         # Assume the risk free rate is zero
#         # sharpe + reward -> HeatMap
#         # Framework -> Models -> Models Performance
#         # 3. Model Design (Optional)
#         # 2. Different Reward
#         # 1. Pretraining + Directly Ranking -> Multiple Objectives
#         # RL Weakness: Hard to generalize to multiple objective
#         # Downside Risk
#         # reward = period_return_rate.mean() / period_return_rate.std()
#         # reward = total_assets - 1
#         # loss = torch.mean(-reward)
#         # reward = total_assets
#         reward = return_rate.mean()
#         loss = -reward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Record Training Information
#         # train_reward += period_return_rate.mean().item() * action.shape[0]
#         train_reward += reward.item()
#         n_samples += action.shape[0]
#     return train_reward / n_samples

# def train_one_epoch_with_costv2(args, train_loader, model, optimizer, device):
#     model.train()
#     train_reward = 0
#     n_samples = 0
#     train_loader, _, _, _ = train_loader
#     transaction_cost_rate = 0.001
#     for batch in train_loader:
#         state, diff, _ = [item.to(device) for item in batch]
#         state = state.squeeze(0)
#         diff = diff.squeeze(0)
#         # action_t-1 + state_t -> action_t
#         # state_t -> action_t  == > action_t-1
#         # action_t-1
#         action = model(state).softmax(-1)
#         batch_size, n_stocks = action.shape
#         # simulate the investment in window size days
#         # total_assets = torch.ones((batch_size,), device=action.device)
#         total_assets = 1
#         for i in range(action.shape[0]):
#             # transaction cost
#             if i == 0:
#                 # window size == total length of training data
#                 # return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
#                 # total_assets = (return_rate + 1) * total_assets
#                 return_rate = ((diff[i, :] - 1 - transaction_cost_rate) * action[i, :]).sum(-1)
#                 total_assets = (return_rate + 1) * total_assets
#                 # reward = (action[:, i, :] * total_assets * diff).sum(-1)
#             else:
#                 # return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
#                 return_rate = ((diff[i, :] - 1) * action[i, :] - (F.relu((action[i, :] - action[i-1, :])) + F.relu(action[i-1, :] - action[i, :])) * transaction_cost_rate).sum(-1)
#                 # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
#                 total_assets = (return_rate + 1) * total_assets
#         reward = total_assets - 1
#         # loss = torch.mean(-reward)
#         loss = -reward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Record Training Information
#         train_reward += -loss.item() * action.shape[0]
#         n_samples += action.shape[0]
#     return train_reward / n_samples

# def train_one_epoch_with_cost(args, train_loader, model, optimizer, device):
#     model.train()
#     train_reward = 0
#     n_samples = 0
#     train_loader, _, _, _ = train_loader
#     transaction_cost_rate = 0.001
#     for batch in train_loader:
#         state, diff, _ = [item.to(device) for item in batch]
#         # action_t-1 + state_t -> action_t
#         # state_t -> action_t  == > action_t-1
#         # action_t-1
#         action = model(state).softmax(-1)
#         batch_size, window, n_stocks = action.shape
#         # simulate the investment in window size days
#         total_assets = torch.ones((batch_size,), device=action.device)
#         for i in range(action.shape[1]):
#             # transaction cost
#             if i == 0:
#                 # window size == total length of training data
#                 return_rate = ((diff[:, i, :] - 1 - transaction_cost_rate) * action[:, i, :]).sum(-1)
#                 total_assets = (return_rate + 1) * total_assets
#                 # reward = (action[:, i, :] * total_assets * diff).sum(-1)
#             else:
#                 return_rate = ((diff[:, i, :] - 1) * action[:, i, :] - (action[:, i, :] - action[:, i-1, :]) * transaction_cost_rate).sum(-1)
#                 # ((action[:, i, :] - action[:, i-1, :]) * total_assets * transaction_cost_rate).sum(-1)
#                 total_assets = (return_rate + 1) * total_assets
#         reward = total_assets - 1
#         loss = torch.mean(-reward)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Record Training Information
#         train_reward += -loss.item() * reward.shape[0]
#         n_samples += reward.shape[0]
#     return train_reward / n_samples

# def train_one_epoch(args, train_loader, model, optimizer, device):
#     model.train()
#     train_reward = 0
#     n_samples = 0
#     train_loader, _, _, _ = train_loader
#     for batch in train_loader:
#         # state, diff, _ = [item.to(device) for item in batch]
#         # state: [1, batch_size, n_stocks, window, n_features]
#         state, diff, _ = [item.to(device) for item in batch]
#         # state: [batch_size, n_stocks, window, n_features]
#         state = state.squeeze(0)
#         # diff: [batch_size, n_stocks]
#         diff = diff.squeeze(0)
#         action = model(state)
#         action = (action / args.temporature).softmax(-1)
#         # action = action.softmax(-1)
#         # action = F.sigmoid(action)
#         # action = action / torch.sum(action, -1, keepdim=True)
#         reward = (action * diff).sum(-1) - 1
#         loss = torch.mean(-reward)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # Record Training Information
#         train_reward += -loss.item() * reward.shape[0]
#         n_samples += reward.shape[0]
#     return train_reward / n_samples

def train_one_epoch(args, train_loader, model, optimizer, device):
    model.train()
    train_reward = 0
    n_samples = 0
    train_loader, _, _, _ = train_loader
    for batch in train_loader:
        # state, diff, _ = [item.to(device) for item in batch]
        # state: [1, batch_size, n_stocks, window, n_features]
        state, diff, _ = [item.to(device) for item in batch]
        # state: [batch_size, n_stocks, window, n_features]
        state = state.squeeze(0)
        # diff: [batch_size, n_stocks]
        diff = diff.squeeze(0)
        # action: [batch_size, n_stocks]
        action = model(state)
        action = (action / args.temporature).softmax(-1)
        # action = action.softmax(-1)
        # action = F.sigmoid(action)
        # action = action / torch.sum(action, -1, keepdim=True)
        reward = (action * diff).sum(-1) - 1
        loss = torch.mean(-reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Record Training Information
        train_reward += -loss.item() * reward.shape[0]
        n_samples += reward.shape[0]
    return train_reward / n_samples

def train_model(args, loader_dict):

    # Set Running Device
    device = torch.device(args.device)
    # TODO: GET DIFFERENT MODEL
    model = getattr(model_factory, args.model_name)(args).to(device)
    # TODO: GET DIFFERNT OPTIMIZER
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = getattr(torch.optim, args.optimizer)(model.out_layer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(model)

    # load pretrained model
    # Need to judge whether to freeze the pretrained weights
    # pretrained_model_path = Path('./pretrained/{}_{}.pth'.format(args.dataset, args.model_name))
    # load_torch_file(model, pretrained_model_path)

    writer = SummaryWriter(comment=f'_{args.model_name}_{args.net_dims}_{args.dataset}_{args.optimizer}_{args.batch_size}_{args.target}_{args.trans_cost}_{args.window}_{args.alpha_downside}_{args.action_constraint}_{args.action_constraint_weight}_{args.mse_alpha}_{args.ranking_alpha}')
    save_path = Path(args.save_dir) / (Path(writer.logdir).stem + '.pth')
    best_eval_total_assets = -1
    best_model = None
    for epoch in range(args.epochs):
        # if args.trans_cost == True:
        #     train_avg_reward = train_one_epoch_with_costv2(args, loader_dict['train'], model, optimizer, device)
        # else:
        # train_avg_reward = train_one_epoch(args, loader_dict['train'], model, optimizer, device)
        # train_avg_reward = train_one_epoch_with_sharpe(args, loader_dict['train'], model, optimizer, device)
        # train_avg_reward = train_one_epoch(args, loader_dict['train'], model, optimizer, device)
        if args.target == 'cumreturn':
            train_avg_reward = train_one_epoch_with_cumulative_return(args, loader_dict['train'], model, optimizer, device)
        elif args.target == 'sharpe':
            train_avg_reward = train_one_epoch_with_sharpe(args, loader_dict['train'], model, optimizer, device)
        # elif args.target == 'onestep':
        #     train_avg_reward = train_one_epoch_with_one_step_reward(args, loader_dict['train'], model, optimizer, device)
        elif args.target == 'downsiderisk':
            train_avg_reward = train_one_epoch_with_downside_risk(args, loader_dict['train'], model, optimizer, device)
        elif args.target == 'actionconstraint':
            train_avg_reward = train_one_epoch_with_action_constraint(args, loader_dict['train'], model, optimizer, device)
        elif args.target == 'cumreturnshort':
            train_avg_reward = train_one_epoch_with_cumulative_return_short(args, loader_dict['train'], model, optimizer, device)
        elif args.target == 'downsideriskqt':
            # train_avg_reward = train_one_epoch_with_downside_risk_dy(args, loader_dict['train'], model, optimizer, device)
            train_avg_reward = train_one_epoch_with_downside_risk_quantile(args, loader_dict['train'], model, optimizer, device)
        else:
            raise ValueError
        if epoch % args.eval_interval == 0:
            if args.trans_cost == True:
                eval_total_assets, _, _, eval_actions = eval_model_with_costv2(args, loader_dict['valid'], model, device)
            else:
                # eval_total_assets, _, _, eval_actions = eval_model_with_action_optimize(args, loader_dict['valid'], model, device)
                eval_total_assets, _, _, eval_actions = eval_model(args, loader_dict['valid'], model, device)
            if eval_total_assets > best_eval_total_assets:
                best_eval_total_assets = eval_total_assets
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), save_path)
            if args.trans_cost == True:
                train_total_assets, _, _, train_actions = eval_model_with_costv2(args, loader_dict['train-test'], model, device)
                test_total_assets, _, _, test_actions = eval_model_with_costv2(args, loader_dict['test'], model, device)
            else:
                # train_total_assets, _, _, train_actions = eval_model_with_action_optimize(args, loader_dict['train-test'], model, device)
                # test_total_assets, _, _, test_actions = eval_model_with_action_optimize(args, loader_dict['test'], model, device)
                train_total_assets, _, _, train_actions = eval_model(args, loader_dict['train-test'], model, device)
                test_total_assets, _, _, test_actions = eval_model(args, loader_dict['test'], model, device)
            # for i in range(len(test_total_assets)):
            #     writer.add_scalar('eval/total_assets_{}'.format(i), eval_total_assets[i], epoch)
            #     writer.add_scalar('test/total_assets_{}'.format(i), test_total_assets[i], epoch)
            writer.add_scalar('eval/total_assets', eval_total_assets, epoch)
            writer.add_scalar('test/total_assets', test_total_assets, epoch)
            if epoch % args.plot_interval == 0:
                eval_actions = eval_actions.detach().cpu().numpy()
                test_actions = test_actions.detach().cpu().numpy()
                train_actions = train_actions[-200:].detach().cpu().numpy()
                eval_df = pd.DataFrame(eval_actions)
                test_df = pd.DataFrame(test_actions)
                train_df = pd.DataFrame(train_actions)
                plt.figure()
                # sns.displot(data=eval_df, x='action', kind='kde')
                sns.heatmap(data=eval_df)
                plt.savefig('./actions_figs/eval/{}_{}_{}_{}_{}_{}.pdf'.format(epoch, args.target, args.batch_size, args.alpha_downside, args.action_constraint, args.action_constraint_weight), bbox_inches='tight')
                plt.figure()
                # sns.displot(data=test_df, x='action', kind='kde')
                sns.heatmap(data=test_df)
                plt.savefig('./actions_figs/test/{}_{}_{}_{}_{}_{}.pdf'.format(epoch, args.target, args.batch_size, args.alpha_downside, args.action_constraint, args.action_constraint_weight), bbox_inches='tight')
                plt.figure()
                sns.heatmap(data=train_df)
                plt.savefig('./actions_figs/train/{}_{}_{}_{}_{}_{}.pdf'.format(epoch, args.target, args.batch_size, args.alpha_downside, args.action_constraint, args.action_constraint_weight), bbox_inches='tight')
        writer.add_scalar('train/train_avg_reward', train_avg_reward, epoch)
        writer.flush()
    return best_model, save_path



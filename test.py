import torch
import json
import pandas as pd
import numpy as np
from pyfolio import timeseries
from pathlib import Path
import torch.nn.functional as F

def eval_model(args, loader, model, device):
    model.eval()
    loader, _, _, _, _ = loader
    with torch.no_grad():
        actions = []
        diffs = []
        idxs = []
        for batch in loader:
            # state: [batch_size, n_stocks, window, n_features]
            state, diff, idx, assets_cov = [item.to(device) for item in batch]
            # action = model(state).softmax(-1)
            # action, _, _ = model(state, assets_cov)
            action, _ = model(state, assets_cov)
            action = action.softmax(-1)
            # action = F.sigmoid(action)
            # action = action / torch.sum(action, -1, keepdim=True)
            actions.append(action)
            diffs.append(diff)
            idxs.append(idx)
        actions = torch.cat(actions)
        diffs = torch.cat(diffs)
        idxs = torch.cat(idxs)
        total_assets = 1
        total_assets_list = [total_assets]
        for i in range(len(actions)):
            # reward = (actions[i] * total_assets * (diffs[i] - 1)).sum() - total_assets
            reward = (actions[i] * total_assets * (diff[i] - 1)).sum() - total_assets
            total_assets += reward
            total_assets_list.append(total_assets.item())
    return total_assets, total_assets_list, idxs, actions

# def eval_model_with_cost(args, loader, model, device, transaction_cost_rate=0.001):
#     model.eval()
#     loader, _, _, _ = loader
#     with torch.no_grad():
#         actions = []
#         diffs = []
#         idxs = []
#         for batch in loader:
#             state, diff, idx = [item.to(device) for item in batch]
#             # action: [batch_size, window, n_stocks]
#             action = model(state).softmax(-1)
#             actions.append(action)
#             diffs.append(diff)
#             idxs.append(idx)
#         actions = torch.cat(actions)[:, -1, :]
#         diffs = torch.cat(diffs)
#         idxs = torch.cat(idxs)
#         total_assets = 1
#         total_assets_list = [total_assets]
#         for i in range(len(actions)):
#             if i == 0:
#                 return_rate = ((diffs[i] - 1 - transaction_cost_rate) * actions[i]).sum(-1)
#             else:
#                 return_rate = ((diffs[i] - 1) * actions[i] - (actions[i] - actions[i-1]) * transaction_cost_rate).sum(-1)
#             total_assets = (return_rate + 1) * total_assets
#             total_assets_list.append(total_assets.item())
#     return total_assets, total_assets_list, idxs, actions

def eval_model_with_costv2(args, loader, model, device, transaction_cost_rate=0.001):
    model.eval()
    loader, _, _, _, _ = loader
    with torch.no_grad():
        actions = []
        diffs = []
        idxs = []
        for batch in loader:
            state, diff, idx, assets_cov = [item.to(device) for item in batch]
            # action: [batch_size, n_stocks]
            action, _ = model(state, assets_cov)
            action = action.softmax(-1)
            actions.append(action)
            diffs.append(diff)
            idxs.append(idx)
        actions = torch.cat(actions)
        diffs = torch.cat(diffs)
        idxs = torch.cat(idxs)
        total_assets = 1
        total_assets_list = [total_assets]
        for i in range(len(actions)):
            if i == 0:
                return_rate = ((diffs[i] - 1 - transaction_cost_rate) * actions[i]).sum(-1)
            else:
                return_rate = ((diffs[i] - 1) * actions[i] - (actions[i] - actions[i-1]) * transaction_cost_rate).sum(-1)
            total_assets = (return_rate + 1) * total_assets
            total_assets_list.append(total_assets.item())
    return total_assets, total_assets_list, idxs, actions

def eval_model_on_metrics(args, loader_dict, model):
    # Set Running Device
    device = torch.device(args.device)
    if args.trans_cost:
        _, total_assets_list, idxs, _ = eval_model_with_costv2(args, loader_dict['test'], model, device)
    else:
        _, total_assets_list, idxs, _ = eval_model(args, loader_dict['test'], model, device)
    # trade_date = [loader_dict['test'][-1][idx] for idx in idxs]
    period_return = pd.Series(np.array(total_assets_list)).pct_change(1).dropna()
    perf_stats_all = timeseries.perf_stats(period_return)
    perf_stats_all_dict = perf_stats_all.to_dict()
    perf_stats_all_dict['cumulative wealth'] = total_assets_list
    # perf_stats_all_dict['date'] = loader_dict['test'][-1][args.window:].tolist()
    result_save_dir = Path(args.result_save_dir) / args.dataset
    if not result_save_dir.exists():
        result_save_dir.mkdir()
    result_save_path = result_save_dir / '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.model_name, args.net_dims, args.dataset, args.optimizer, args.batch_size, args.target, args.trans_cost, args.window, args.alpha_downside, args.action_constraint, args.action_constraint_weight, args.mse_alpha, args.ranking_alpha)
    with open(result_save_path, 'w') as f:
        json.dump(perf_stats_all_dict, f, indent=4)

def eval_model_with_action_optimize(args, loader, model, device):
    model.eval()
    loader, _, _, _, all_cov = loader
    with torch.no_grad():
        actions_no_softmax = []
        actions = []
        diffs = []
        idxs = []
        for batch in loader:
            # state: [batch_size, n_stocks, window, n_features]
            state, diff, idx, assets_cov = [item.to(device) for item in batch]
            # action = model(state).softmax(-1)
            action, _ = model(state, assets_cov)
            actions_no_softmax.append(action)
            action = action.softmax(-1)
            # action = F.sigmoid(action)
            # action = action / torch.sum(action, -1, keepdim=True)
            actions.append(action)
            diffs.append(diff)
            idxs.append(idx)
        actions = torch.cat(actions)
        diffs = torch.cat(diffs)
        idxs = torch.cat(idxs)
        total_assets = 1
        total_assets_list = [total_assets]
        for i in range(len(actions)):
            reward = (actions[i] * total_assets * diffs[i]).sum() - total_assets
            total_assets += reward
            total_assets_list.append(total_assets.item())
    # Action Optimize
    optimize_total_assets = [total_assets]
    all_cov = all_cov.to(device)
    actions = torch.cat(actions_no_softmax)
    actions.requires_grad = True
    for i in range(args.n_optimize):
        actions = actions.detach()
        actions.requires_grad = True
        temp_actions = actions.softmax(-1)
        cov_constraint = temp_actions.unsqueeze(1) @ all_cov.detach() @ temp_actions.unsqueeze(-1)
        loss = (cov_constraint - args.predefined_cov_threshold).sum()
        loss.backward()
        actions = actions - 100 * actions.grad
        # actions.grad = torch.zeros_like(actions.grad)
        total_assets = 1
        # temp_total_assets_list = [total_assets]
        temp_actions = actions.softmax(-1)
        for i in range(len(actions)):
            reward = (temp_actions[i] * total_assets * diffs[i]).sum() - total_assets
            total_assets += reward
            # temp_total_assets_list.append(total_assets.item())
        optimize_total_assets.append(total_assets)
    return optimize_total_assets, total_assets_list, idxs, actions.softmax(-1)

def eval_model_on_metrics_with_action_optimize(args, loader_dict, model):
    # Set Running Device
    device = torch.device(args.device)
    if args.trans_cost:
        _, total_assets_list, idxs, _ = eval_model_with_costv2(args, loader_dict['test'], model, device)
    else:
        _, total_assets_list, idxs, _ = eval_model(args, loader_dict['test'], model, device)
    # trade_date = [loader_dict['test'][-1][idx] for idx in idxs]
    period_return = pd.Series(np.array(total_assets_list)).pct_change(1).dropna()
    perf_stats_all = timeseries.perf_stats(period_return)
    perf_stats_all_dict = perf_stats_all.to_dict()
    perf_stats_all_dict['cumulative wealth'] = total_assets_list
    perf_stats_all_dict['date'] = loader_dict['test'][-1][args.window:].tolist()
    result_save_dir = Path(args.result_save_dir) / args.dataset
    if not result_save_dir.exists():
        result_save_dir.mkdir()
    result_save_path = result_save_dir / '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.model_name, args.net_dims, args.dataset, args.optimizer, args.batch_size, args.target, args.trans_cost, args.window, args.alpha_downside, args.action_constraint, args.action_constraint_weight, args.mse_alpha, args.ranking_alpha)
    with open(result_save_path, 'w') as f:
        json.dump(perf_stats_all_dict, f, indent=4)
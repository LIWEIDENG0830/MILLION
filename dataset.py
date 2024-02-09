import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from pathlib import Path
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def load_data_from_whole_csv(args):
    dataset_path = Path(args.data_dir) / 'processed.csv'
    data_df = pd.read_csv(dataset_path, index_col=0)
    return data_df

def data_split_from_whole(data_df, args):
    train_df = data_split(data_df, start=args.train_start_date, end=args.train_end_date)
    valid_df = data_split(data_df, start=args.valid_start_date, end=args.valid_end_date)
    test_df = data_split(data_df, start=args.test_start_date, end=args.test_end_date)
    return train_df, valid_df, test_df

class NewDataset:
    # should set the batch_size to 1
    def __init__(self, all_state, diff_close, idx_map, all_cov, batch_size):
        super().__init__()
        # all_state: [n_samples, n_stocks, window, n_features]
        self.all_state = all_state
        # diff_close: [n_samples, n_stocks]
        self.diff_close = diff_close
        self.idx_map = idx_map
        self.all_cov = all_cov
        self.batch_size = batch_size
        self.length = len(all_state)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # item_state: [batch_size, n_stocks, window, n_features]
        item_state = self.all_state[idx: idx+self.batch_size]
        # item_diff_close: [batch_size, n_stocks]
        item_diff_close = self.diff_close[idx: idx+self.batch_size]
        item_idx_map = self.idx_map[idx: idx+self.batch_size]
        item_cov = self.all_cov[idx: idx+self.batch_size]
        return item_state, item_diff_close, item_idx_map, item_cov
    
    def get_batches(self):
        # randomly select the start time
        # start, start + batch_size ; 
        start = random.randint(0, self.batch_size-1)
        idx = start
        while idx + self.batch_size < self.length:
            item_state = self.all_state[idx: idx+self.batch_size]
            item_diff_close = self.diff_close[idx: idx+self.batch_size]
            item_idx_map = self.idx_map[idx: idx+self.batch_size]
            item_cov = self.all_cov[idx: idx+self.batch_size]
            yield item_state, item_diff_close, item_idx_map, item_cov
            # Nonoverlap
            idx += self.batch_size

class DataProcessor(object):
    def __init__(self, args):
        self.args = args
        self.npz_file = Path(args.data_dir) / args.npz_file
        # self.indicators = ['zopen', 'zhigh', 'zlow', 'zclose', 'zd_5', 'zd_10', 'zd_15', 'zd_20', 'zd_25', 'zd_30']
        self.indicators = INDICATORS
        
        # if not npz_file.exists():
        #     print('Preparing Data From Scratch')
        #     self.close_dict, self.tech_dict = self.from_scratch(args, npz_file)
        # else:
        #     print('Preparing Data From Dumped Data')
        #     self.close_dict, self.tech_dict = self.from_dumped(npz_file)
        self.close_dict, self.tech_dict, self.date_dict, self.cov_dict = self.from_scratch(args, None)
    
    def set_data_args(self):
        self.args.n_stocks = len(self.tic_list)
        # self.args.state_dim = len(INDICATORS)
        self.args.state_dim = len(self.indicators)
        self.args.dataset = self.npz_file.parent.stem
    
    def from_dumped(self, npz_file):
        ary_dict = np.load(npz_file, allow_pickle=True)
        close_arys = ary_dict['close_ary']
        tech_arys = ary_dict['tech_ary']
        date_arys = ary_dict['date_ary']
        close_dict = {'train': close_arys[0], 'valid': close_arys[1], 'test': close_arys[2]}
        tech_dict = {'train': tech_arys[0], 'valid': tech_arys[1], 'test': tech_arys[2]}
        date_dict = {'train': date_arys[0], 'valid': date_arys[1], 'test': date_arys[2]}
        return close_dict, tech_dict, date_dict
    
    def from_scratch(self, args, npz_file):
        ''' Load Data '''
        data_df = load_data_from_whole_csv(args)
        subdata_df_list = []
        for tic, subdata_df in data_df.groupby('tic'):
            # subdata_df[INDICATORS] = subdata_df[INDICATORS] / subdata_df[INDICATORS].max()
            # subdata_df[INDICATORS] = (subdata_df[INDICATORS] - subdata_df[INDICATORS].mean()) / subdata_df[INDICATORS].std()
            # subdata_df['zopen'] = (subdata_df['open'] / subdata_df['close']) - 1
            # subdata_df['zhigh'] = (subdata_df['high'] / subdata_df['close']) - 1
            # subdata_df['zlow'] = (subdata_df['low'] / subdata_df['close']) - 1
            # subdata_df['zclose'] = subdata_df['close'].pct_change(1)
            # subdata_df['zd_5'] = (subdata_df['close'].rolling(5).mean() / subdata_df['close']) - 1
            # subdata_df['zd_10'] = (subdata_df['close'].rolling(10).mean() / subdata_df['close']) - 1
            # subdata_df['zd_15'] = (subdata_df['close'].rolling(15).mean() / subdata_df['close']) - 1
            # subdata_df['zd_20'] = (subdata_df['close'].rolling(20).mean() / subdata_df['close']) - 1
            # subdata_df['zd_25'] = (subdata_df['close'].rolling(25).mean() / subdata_df['close']) - 1
            # subdata_df['zd_30'] = (subdata_df['close'].rolling(30).mean() / subdata_df['close']) - 1
            subdata_df[self.indicators] = (subdata_df[self.indicators] - subdata_df[self.indicators].mean()) / subdata_df[self.indicators].std()
            subdata_df_list.append(subdata_df[['date', 'tic', 'close'] + self.indicators])
            # subdata_df_list.append(subdata_df)
        subdata_df = pd.concat(subdata_df_list).dropna()
        data_df = subdata_df.sort_values(by=['date', 'tic'])
        data_df.index = data_df.date.factorize()[0]
        # Compute Cov
        lookback = 20
        return_list = []
        cov_list = []
        for i in tqdm(range(lookback,len(data_df.index.unique()))):
            data_lookback = data_df.loc[i-lookback:i,:]
            price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
            return_lookback = price_lookback.pct_change(fill_method=None).dropna()
            return_list.append(return_lookback)
            covs = return_lookback.cov().values 
            cov_list.append(covs)
        df_cov = pd.DataFrame({'date':data_df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
        # data_df = data_df.merge(df_cov, on='date')
        data_df = data_df.sort_values(['date','tic']).reset_index(drop=True)
        # data_df[INDICATORS] = (data_df[INDICATORS] - data_df[INDICATORS].mean()) / data_df[INDICATORS].std()
        # TODO: Whether needs to preprocess before data split
        ''' Split Data '''
        train_df, valid_df, test_df = data_split_from_whole(data_df, args)
        ''' Extract information from train/valid/test df '''
        tech_arys = []
        close_arys = []
        date_arys = []
        cov_arys = []
        self.tic_list = train_df.tic.unique()
        df_list = [train_df, valid_df, test_df]
        for data_df in df_list:
            tech_ary = []
            close_ary = []
            for tic in self.tic_list:
                tic_df = data_df[data_df['tic'] == tic]
                # tech_ary.append(tic_df[INDICATORS].values)
                tech_ary.append(tic_df[self.indicators].values)
                close_ary.append(tic_df['close'].values)
            # n_timestamps, n_stocks, n_techs
            tech_ary = np.stack(tech_ary, 1)
            # n_timestamps, n_stocks
            close_ary = np.stack(close_ary, 1)
            tech_arys.append(tech_ary)
            close_arys.append(close_ary)
            date_arys.append(data_df.date.unique())
            # import pdb; pdb.set_trace()
            cov_arys.append(df_cov[(df_cov['date'] <= data_df['date'].iloc[-1]) & (df_cov['date'] >= data_df['date'].iloc[0])]['cov_list'].values)
        if npz_file is not None:
            np.savez_compressed(npz_file, close_ary=close_arys, tech_ary=tech_arys, date_ary=date_arys)
        close_dict = {'train': close_arys[0], 'valid': close_arys[1], 'test': close_arys[2]}
        tech_dict = {'train': tech_arys[0], 'valid': tech_arys[1], 'test': tech_arys[2]}
        date_dict = {'train': date_arys[0], 'valid': date_arys[1], 'test': date_arys[2]}
        cov_dict = {'train': cov_arys[0], 'valid': cov_arys[1], 'test': cov_arys[2]}
        return close_dict, tech_dict, date_dict, cov_dict
    
def get_loader(args, tech_ary, close_ary, cov_ary, key):
    window = args.window
    batch_size = args.batch_size
    n_stocks = args.n_stocks

    all_state = []
    diff_close = []
    all_cov = []
    idx_map = []
    max_step = len(close_ary)
    for i in range(max_step-window):
        all_state.append(torch.tensor(tech_ary[i: i+window].reshape(window, n_stocks, -1).transpose(1,0,2), dtype=torch.float32))
        diff_close.append(torch.tensor(close_ary[i+window] / close_ary[i+window-1], dtype=torch.float32))
        # TODO: BUG all_cov
        all_cov.append(torch.tensor(cov_ary[i+window-1], dtype=torch.float32))
        # all_cov.append(torch.tensor(close_ary[i+window-1], dtype=torch.float32))
        idx_map.append(torch.tensor([i+window-1], dtype=torch.long))
    # all_state: [n_samples, n_stocks, window, n_features]
    all_state = torch.stack(all_state, 0)
    # diff_close: [n_samples, n_stocks]
    diff_close = torch.stack(diff_close, 0)
    # all_cov: [n_samples, n_stocks, n_stocks]
    all_cov = torch.stack(all_cov, 0)
    idx_map = torch.cat(idx_map, 0)

    # train_set = TensorDataset(all_state, diff_close, idx_map)
    if key == 'train':
        # train_set = NewDataset(all_state, diff_close, idx_map, batch_size)
        # train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        train_set = NewDataset(all_state, diff_close, idx_map, all_cov, batch_size)
        train_loader = train_set
    else:
        train_set = TensorDataset(all_state, diff_close, idx_map, all_cov)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    return train_loader, all_state, diff_close, all_cov

# def get_loader(args, tech_ary, close_ary, shuffle):
#     window = args.window
#     batch_size = args.batch_size
#     n_stocks = args.n_stocks

#     all_state = []
#     diff_close = []
#     idx_map = []
#     max_step = len(close_ary)
#     for i in range(max_step-window):
#         all_state.append(torch.tensor(tech_ary[i: i+window].reshape(window, n_stocks, -1).transpose(1,0,2), dtype=torch.float32))
#         diff_close.append(torch.tensor(close_ary[i+window] / close_ary[i+window-1]))
#         idx_map.append(torch.tensor([i+window-1], dtype=torch.long))
#     # all_state: [n_samples, n_stocks, window, n_features]
#     all_state = torch.stack(all_state, 0)
#     # diff_close: [n_samples, n_stocks]
#     diff_close = torch.stack(diff_close, 0)
#     idx_map = torch.cat(idx_map, 0)

#     train_set = TensorDataset(all_state, diff_close, idx_map)
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
#     return train_loader, all_state, diff_close

# def get_loader_transaction_cost(args, tech_ary, close_ary, shuffle):
#     window = args.window
#     batch_size = args.batch_size
#     n_stocks = args.n_stocks

#     all_state = []
#     diff_close = []
#     idx_map = []
#     max_step = len(close_ary)
#     for i in range(max_step-window):
#         all_state.append(torch.tensor(tech_ary[i: i+window].reshape(window, n_stocks, -1).transpose(1,0,2), dtype=torch.float32))
#         diff_close.append(torch.tensor(close_ary[i+1:i+window+1] / close_ary[i:i+window]))
#         idx_map.append(torch.tensor([i+window-1], dtype=torch.long))
#     # all_state: [n_samples, n_stocks, window, n_features]
#     all_state = torch.stack(all_state, 0)
#     # diff_close: [n_samples, window, n_stocks]
#     diff_close = torch.stack(diff_close, 0)
#     idx_map = torch.cat(idx_map, 0)

#     train_set = TensorDataset(all_state, diff_close, idx_map)
#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=shuffle)
#     return train_loader, all_state, diff_close

def get_loaders(args, tech_dict, close_dict, date_dict, cov_dict):
    loader_dict = {}
    for key in tech_dict:
        # if not args.trans_cost:
        #     if key == 'train':
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=True)
        #     else:
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        #     loader_dict[key] = [loader, all_state, diff_close, date_dict[key]]
        # else:
        #     if key == 'train':
        #         # loader, all_state, diff_close = get_loader_transaction_cost(args, tech_dict[key], close_dict[key], shuffle=True)
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        #     else:
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        #     loader_dict[key] = [loader, all_state, diff_close, date_dict[key]]
        loader, all_state, diff_close, all_cov = get_loader(args, tech_dict[key], close_dict[key], cov_dict[key], key)
        # loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        loader_dict[key] = [loader, all_state, diff_close, date_dict[key], all_cov]
    # This is used for action testing
    loader, all_state, diff_close, all_cov = get_loader(args, tech_dict['train'], close_dict['train'], cov_dict['train'], key='train-test')
    loader_dict['train-test'] = [loader, all_state, diff_close, close_dict['train'], all_cov]
    return loader_dict


def get_loaders_pretrain(args, tech_dict, close_dict, date_dict):
    loader_dict = {}
    for key in tech_dict:
        # if not args.trans_cost:
        #     if key == 'train':
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=True)
        #     else:
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        #     loader_dict[key] = [loader, all_state, diff_close, date_dict[key]]
        # else:
        #     if key == 'train':
        #         # loader, all_state, diff_close = get_loader_transaction_cost(args, tech_dict[key], close_dict[key], shuffle=True)
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        #     else:
        #         loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        #     loader_dict[key] = [loader, all_state, diff_close, date_dict[key]]
        loader, all_state, diff_close, all_cov = get_loader(args, tech_dict[key], close_dict[key], 'test')
        # loader, all_state, diff_close = get_loader(args, tech_dict[key], close_dict[key], shuffle=False)
        loader_dict[key] = [loader, all_state, diff_close, date_dict[key], all_cov]
    # This is used for action testing
    loader, all_state, diff_close, all_cov = get_loader(args, tech_dict['train'], close_dict['train'], key='train-test')
    loader_dict['train-test'] = [loader, all_state, diff_close, close_dict['train']]
    return loader_dict

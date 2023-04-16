import datetime
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from config import const
import numpy as np


rec_inter_history = pickle.load(open(const.recommendation_index, 'rb'))
search_inter_history = pickle.load(open(const.search_index, 'rb'))
open_search_inter_history = pickle.load(open(const.open_search_index, 'rb'))

search_vocab = pickle.load(open(const.search_vocab,'rb'))
reco_vocab = pickle.load(open(const.reco_vocab,'rb'))
user_vocab = pickle.load(open(const.user_vocab, 'rb'))



"""
add user info
"""

"""
add repeat network
"""

class MyDataset(Dataset):
    def __init__(self, main_file_path, max_len_reco, max_len_search, max_len_open_search):
        self.df = pd.read_csv(main_file_path, sep='\t')
        self.rec_inter_history = rec_inter_history
        self.search_inter_history = search_inter_history
        self.open_search_inter_history = open_search_inter_history

        self.max_len_rec = max_len_reco
        self.max_len_search = max_len_search
        self.max_len_open_search = max_len_open_search


    def __len__(self):
        return len(self.df)
    def get_time_feature(self, time):
        timestamp = time / 1000
        dt = datetime.datetime.fromtimestamp(timestamp)
        hour_sin = np.sin(2 * np.pi * dt.hour / 24, dtype=np.float32)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24, dtype=np.float32)
        minute_sin = np.sin(2 * np.pi * dt.minute / 60, dtype=np.float32)
        minute_cos = np.cos(2 * np.pi * dt.minute / 60, dtype=np.float32)
        second_sin = np.sin(2 * np.pi * dt.second / 60, dtype=np.float32)
        second_cos = np.cos(2 * np.pi * dt.second / 60, dtype=np.float32)

        time_features = [hour_sin, hour_cos, minute_sin, minute_cos, second_sin, second_cos]

        return time_features        

    def __getitem__(self, idx):
        user_id, time, label = self.df.loc[idx, ['user_id', 'request_time_ms', 'label']]
        time_features = self.get_time_feature(time)
        rec_inter_history_s = [reco_vocab[i] for i in self.rec_inter_history[(user_id, time)][0]]
        search_inter_history_s = [search_vocab[i] for i in self.search_inter_history[(user_id, time)][0]]
        open_search_inter_history_s = [search_vocab[i] for i in self.open_search_inter_history[(user_id, time)][0]]
        rec_inter_time_s = [self.get_time_feature(i) for i in self.rec_inter_history[(user_id, time)][1]]
        search_inter_time_s = [self.get_time_feature(i) for i in self.search_inter_history[(user_id, time)][1]]
        open_search_inter_time_s = [self.get_time_feature(i) for i in self.open_search_inter_history[(user_id, time)][1]]

        rec_inter_history_s = rec_inter_history_s[-self.max_len_rec:] if len(rec_inter_history_s) > self.max_len_rec else [0] * (self.max_len_rec - len(rec_inter_history_s)) + rec_inter_history_s
        search_inter_history_s = search_inter_history_s[-self.max_len_search:] if len(search_inter_history_s) > self.max_len_search else [0] * (self.max_len_search - len(search_inter_history_s)) + search_inter_history_s
        open_search_inter_history_s = open_search_inter_history_s[-self.max_len_open_search:] if len(open_search_inter_history_s) > self.max_len_open_search else [0] * (self.max_len_open_search - len(open_search_inter_history_s)) + open_search_inter_history_s

        rec_inter_time_s = rec_inter_time_s[-self.max_len_rec:] if len(rec_inter_time_s) > self.max_len_rec else [[0 for _ in range(len(time_features))]] * (self.max_len_rec - len(rec_inter_time_s)) + rec_inter_time_s
        search_inter_time_s = search_inter_time_s[-self.max_len_search:] if len(search_inter_time_s) > self.max_len_search else [[0 for _ in range(len(time_features))]] * (self.max_len_search - len(search_inter_time_s)) + search_inter_time_s
        open_search_inter_time_s = open_search_inter_time_s[-self.max_len_open_search:] if len(open_search_inter_time_s) > self.max_len_open_search else [[0 for _ in range(len(time_features))]] * (self.max_len_open_search - len(open_search_inter_time_s)) + open_search_inter_time_s
        
        return rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_vocab[user_id],label,rec_inter_time_s,search_inter_time_s, open_search_inter_time_s # label这块将reco设置为了1，search设置为了0其实是反的，所以这里的label取反了


def my_collate_fn(batch):
    # Extract the elements from the batch
    rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label, rec_inter_time_s,search_inter_time_s, open_search_inter_time_s = zip(*batch)
    
    # Pad the sequences to the same length
    rec_inter_history_s = torch.tensor(rec_inter_history_s)
    
    search_inter_history_s = torch.tensor(search_inter_history_s)
    
    open_search_inter_history_s = torch.tensor(open_search_inter_history_s)

    time_features = torch.tensor(time_features, dtype=torch.float)

    user_id = torch.tensor(user_id)
    
    label = torch.tensor(label)
    rec_inter_time_s = torch.tensor(rec_inter_time_s)
    search_inter_time_s = torch.tensor(search_inter_time_s)
    open_search_inter_time_s = torch.tensor(open_search_inter_time_s)
    
    return rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, time_features, user_id, label,rec_inter_time_s, search_inter_time_s, open_search_inter_time_s

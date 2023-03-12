import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from config import const

rec_inter_history = pickle.load(open(const.recommendation_index, 'rb'))
search_inter_history = pickle.load(open(const.search_index, 'rb'))
open_search_inter_history = pickle.load(open(const.open_search_index, 'rb'))


"""
加载一下vocab 将history中的东西转化为id
"""

class MyDataset(Dataset):
    def __init__(self, main_file_path, max_len=10):
        self.df = pd.read_csv(main_file_path, sep='\t')
        self.rec_inter_history = rec_inter_history
        self.search_inter_history = search_inter_history
        self.open_search_inter_history = open_search_inter_history

        self.max_len_rec = max_len+10
        self.max_len_search = max_len
        self.max_len_open_search = max_len-5


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id, time, label = self.df.loc[idx, ['user_id', 'time', 'label']]
        rec_inter_history_s = self.rec_inter_history[(user_id, time)]
        search_inter_history_s = self.search_inter_history[(user_id, time)]
        open_search_inter_history_s = self.open_search_inter_history[(user_id, time)]
        rec_inter_history_s = rec_inter_history_s[-self.max_len_rec:] if len(rec_inter_history_s) > self.max_len_rec else [0] * (self.max_len_rec - len(rec_inter_history_s)) + rec_inter_history_s
        search_inter_history_s = search_inter_history_s[-self.max_len_search:] if len(search_inter_history_s) > self.max_len_search else [0] * (self.max_len_search - len(search_inter_history_s)) + search_inter_history_s
        open_search_inter_history_s = open_search_inter_history_s[-self.max_len_open_search:] if len(open_search_inter_history_s) > self.max_len_open_search else [0] * (self.max_len_open_search - len(open_search_inter_history_s)) + open_search_inter_history_s

        return rec_inter_history_s, search_inter_history_s, open_search_inter_history_s, label



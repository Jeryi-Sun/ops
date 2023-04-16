import torch
from torch import nn
import torch.nn.functional as F
from .transformer import ModelArgs, Transformer
from .utils import FocalLoss


class basic_model(nn.Module):
    def __init__(self, params_reco:ModelArgs, params_search:ModelArgs, params_open_search:ModelArgs):
        super().__init__()
        self.reco_embedding =  nn.Embedding(
            params_reco.vocab_size, params_reco.embedding_dim
        )
        self.search_embedding = nn.Embedding(
            params_search.vocab_size, params_search.embedding_dim
        )
        self.user_embedding = nn.Embedding(
            ModelArgs.user_num, ModelArgs.dim
        )
        self.reco_tf = Transformer(params=params_reco, embedding=self.reco_embedding)
        self.search_tf = Transformer(params=params_search,embedding=self.search_embedding)
        self.open_search_tf = Transformer(params=params_open_search,embedding=self.search_embedding)
        self.output_linear= nn.Sequential(nn.Linear(
             params_reco.dim*4+6, params_reco.dim
         ), nn.LeakyReLU(),
         nn.Linear(
             params_reco.dim, 2
         ))
        weights = torch.tensor([1, 9], dtype=torch.float32)
        self.loss_func = nn.CrossEntropyLoss(weight=weights)
        #self.loss_func = FocalLoss()


    def forward(self, reco_history, search_history, open_search_history, time_features, user_id, reco_time, search_time, open_search_time):
       output_reco = self.reco_tf(reco_history,transformer_time_feature=reco_time)
       output_search = self.search_tf(search_history, transformer_time_feature=search_time)
       output_open_search = self.open_search_tf(open_search_history, transformer_time_feature=open_search_time)
       user_feature = self.user_embedding(user_id)
       return self.output_linear(torch.cat([output_reco, output_search, output_open_search, user_feature, time_features], dim=-1))
    def train_(self, reco_history, search_history, open_search_history, time_features, user_id, label, rec_inter_time_s, search_inter_time_s, open_search_inter_time_s):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id, rec_inter_time_s, search_inter_time_s, open_search_inter_time_s)
        return self.loss_func(output, label)
    @torch.inference_mode()
    def infer_(self, reco_history, search_history, open_search_history, time_features, user_id,rec_inter_time_s, search_inter_time_s, open_search_inter_time_s ):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id, rec_inter_time_s, search_inter_time_s, open_search_inter_time_s)
        return output

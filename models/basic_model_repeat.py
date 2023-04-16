import torch
from torch import nn
import torch.nn.functional as F
from .transformer import ModelArgs, Transformer
from .target_attention import TargetAttention
from .utils import FocalLoss


class basic_model_repeat(nn.Module):
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
        self.linear1 = nn.Sequential(nn.Linear(
             params_reco.dim*3, params_reco.dim*2
         ), nn.LeakyReLU(),
         nn.Linear(
             params_reco.dim*2, params_reco.dim
         ), nn.LeakyReLU())

        self.output_linear= nn.Sequential(nn.Linear(
             params_reco.dim*3+6, params_reco.dim*3
         ), nn.LeakyReLU(),
         nn.Linear(
             params_reco.dim*3, params_reco.dim
         ), nn.LeakyReLU(),
         nn.Linear(
             params_reco.dim, 2
         ))
        weights = torch.tensor([1, 9], dtype=torch.float32)
        self.loss_func = nn.CrossEntropyLoss()
        #self.loss_func = FocalLoss()
        self.repeat_model = TargetAttention(3*params_reco.dim, 2*params_reco.dim, params_reco.dim)

    def forward(self, reco_history, search_history, open_search_history, time_features, user_id):
       output_reco = self.reco_tf(reco_history)
       output_search = self.search_tf(search_history)
       output_open_search = self.open_search_tf(open_search_history)
       user_feature = self.user_embedding(user_id)
       output_repeat = self.repeat_model(torch.cat([output_reco, output_search, output_open_search], dim=-1), self.search_embedding(search_history))[0]
       history_feature = self.linear1(torch.cat([output_reco, output_search, output_open_search], dim=-1))
       return self.output_linear(torch.cat([history_feature, user_feature,output_repeat,time_features], dim=-1))
       
    def train_(self, reco_history, search_history, open_search_history, time_features, user_id, label):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id)
        return self.loss_func(output, label)
    @torch.inference_mode()
    def infer_(self, reco_history, search_history, open_search_history, time_features, user_id ):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id)
        return output

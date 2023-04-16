import torch
from torch import nn
import torch.nn.functional as F
from .transformer import ModelArgs, Transformer
from .utils import FocalLoss


class basic_model_mean(nn.Module):
    def __init__(self, params_reco:ModelArgs, params_search:ModelArgs, params_open_search:ModelArgs):
        super().__init__()
        self.reco_embedding =  nn.Embedding(
            params_reco.vocab_size, params_reco.dim
        )
        self.search_embedding = nn.Embedding(
            params_search.vocab_size, params_search.dim
        )
        self.user_embedding = nn.Embedding(
            ModelArgs.user_num, ModelArgs.dim
        )
        self.output_linear= nn.Sequential(nn.Linear(
             params_reco.dim*4+6, params_reco.dim
         ), nn.LeakyReLU(),
         nn.Linear(
             params_reco.dim, 2
         ))
        weights = torch.tensor([1, 9], dtype=torch.float32)
        self.loss_func = nn.CrossEntropyLoss()
        #self.loss_func = FocalLoss()


    def forward(self, reco_history, search_history, open_search_history, time_features, user_id):
       output_reco = torch.mean(self.reco_embedding(reco_history), dim=1)
       output_search = torch.mean(self.search_embedding(search_history), dim=1)
       output_open_search = torch.mean(self.search_embedding(open_search_history), dim=1)
       user_feature = self.user_embedding(user_id)
       return self.output_linear(torch.cat([output_reco, output_search, output_open_search, user_feature, time_features], dim=-1))
    def train_(self, reco_history, search_history, open_search_history, time_features, user_id, label):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id)
        return self.loss_func(output, label)
    @torch.inference_mode()
    def infer_(self, reco_history, search_history, open_search_history, time_features, user_id ):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id)
        return output

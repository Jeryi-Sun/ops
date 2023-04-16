import torch
from torch import nn
import torch.nn.functional as F
from .transformer import ModelArgs, Transformer
from .utils import FocalLoss

def weighted_average(x):
    """
    对序列x(batch_size, seq, hidden_size)进行加权平均，
    使得排序越靠后的权重越大
    """
    # 计算排序权重
    _, seq_len, _ = x.size()
    weights = torch.arange(seq_len, 0, -1, dtype=torch.float32, device=x.device)
    weights = torch.nn.functional.softmax(weights, dim=-1)
    
    # 对序列进行排序并进行加权平均
    sorted_x, _ = torch.sort(x, dim=1, descending=True)
    weighted_x = sorted_x * weights.view(1, -1, 1)
    output = torch.sum(weighted_x, dim=1)
    return output

class basic_model_weight_mean(nn.Module):
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
       output_reco = weighted_average(self.reco_embedding(reco_history))
       output_search = weighted_average(self.search_embedding(search_history))
       output_open_search = weighted_average(self.search_embedding(open_search_history))
       user_feature = self.user_embedding(user_id)
       return self.output_linear(torch.cat([output_reco, output_search, output_open_search, user_feature, time_features], dim=-1))
    def train_(self, reco_history, search_history, open_search_history, time_features, user_id, label):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id)
        return self.loss_func(output, label)
    @torch.inference_mode()
    def infer_(self, reco_history, search_history, open_search_history, time_features, user_id ):
        output = self.forward(reco_history, search_history, open_search_history,time_features, user_id)
        return output

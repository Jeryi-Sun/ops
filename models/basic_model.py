import torch
from torch import nn
import torch.nn.functional as F
from transformer import ModelArgs, Transformer

"""
后面接上linear
"""
class basic_model(nn.Module):
    def __init__(self, params_reco:ModelArgs, params_search:ModelArgs, params_open_search:ModelArgs):
        super().__init__()
        self.reco_tf = Transformer(params=params_reco)
        self.search_tf = Transformer(params=params_search)
        self.open_search_tf = Transformer(params=params_open_search)


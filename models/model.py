import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.positional_encoding = self._get_positional_encoding()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([self._get_layer() for _ in range(num_layers)])

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        inputs = self.embedding(inputs) * torch.sqrt(self.hidden_dim)
        inputs += self.positional_encoding[:, :seq_len, :].to(inputs.device)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def _get_positional_encoding(self):
        pos = torch.arange(0, 1000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * -(math.log(10000.0) / self.hidden_dim))
        sin = torch.sin(pos * div_term)
        cos = torch.cos(pos * div_term)
        pos_encoding = torch.cat([sin, cos], dim=-1)
        return pos_encoding.unsqueeze(0)

    def _get_layer(self):
        return nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.num_heads)

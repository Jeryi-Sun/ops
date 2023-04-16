import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetAttention(nn.Module):
    def __init__(self, hidden_size, ffn_size, output_size):
        super(TargetAttention, self).__init__()
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Linear(ffn_size, output_size)
        )

    def forward(self, t, h):
        # Apply the FFN to t
        t_ffn = self.ffn(t)

        # Compute attention scores
        t_ffn = t_ffn.unsqueeze(1)
        attention_scores = torch.bmm(t_ffn, h.transpose(1, 2))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum of h using attention_weights
        context = torch.bmm(attention_weights, h)

        # Squeeze the context to remove the extra dimension
        context = context.squeeze(1)

        return context, attention_weights
if __name__ == '__main__':
    # Initialize the TargetAttention model
    hidden_size = 128
    ffn_size = 256
    model = TargetAttention(hidden_size, ffn_size)

    # Dummy input tensors
    batch_size = 32
    sequence_len = 10
    t = torch.randn(batch_size, hidden_size)
    h = torch.randn(batch_size, sequence_len, hidden_size)

    # Forward pass
    context, attention_weights = model(t, h)
    print(context)

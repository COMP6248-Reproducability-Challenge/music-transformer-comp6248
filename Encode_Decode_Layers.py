import torch
import torch.nn as nn
from Sublayers import MultiHeadAttention, Norm
from PositionalFeedForward import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff = 1024, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)  # create normalisation sublayer with size d_model
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        z = self.dropout_1(self.attn(x, x, x, mask))  # attention layer 1 with dropout_1
        x = self.norm_1(x + z)  # add and normalise

        z = self.dropout_2(self.ff(x))  # attention layer 2 with dropout_2
        x = self.norm_2(x + z)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff = 1024, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
# We can then build a convenient cloning function that can generate multiple layers:
    def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

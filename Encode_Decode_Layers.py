import torch
import torch.nn as nn
from Sublayers import MultiHeadAttention, Norm
from PositionalFeedForward import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff = 1024, dropout = 0.0, attention_type = "Baseline"):
        super().__init__()
        self.norm_1 = Norm(d_model)  # create normalisation sublayer with size d_model
        self.norm_2 = Norm(d_model)
        self.attention_type = attention_type

        self.attn = MultiHeadAttention(heads, d_model, attention_type = self.attention_type)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff = 1024, dropout=0.0, attention_type = "Baseline"):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.attention_type = attention_type
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, attention_type = self.attention_type)
        self.attn_2 = MultiHeadAttention(heads, d_model, attention_type = self.attention_type)
        self.ff = FeedForward(d_model, d_ff, dropout)
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))

        # x2 = self.norm_1(x + self.dropout_1(self.attn_1(x, x, x, trg_mask)))
        # x2 = self.norm_2(x2 + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask)))
        # x2 = self.norm_3(x2 + self.dropout_3(self.ff(x2)))

        return x
# We can then build a convenient cloning function that can generate multiple layers:
    def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

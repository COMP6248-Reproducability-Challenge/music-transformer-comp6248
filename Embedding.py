# Filename: Embedding.py
# Date Created: 08-Mar-2019 4:38:59 pm
# Author: zckoh
# Description: Embedding method before input to encoder.
#              Includes basic embedding, positional encoding,
#              and concatenating positional encoding.
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    """
        vocab_size = size of the dictionary of embeddings
        d_model = size of each embedding vectors
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # make embeddings relatively larger
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoder(nn.Module):
    "Implement the PE function with addition."
    def __init__(self, d_model, dropout = 0.1, max_seq_len = 1024):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class PositionalEncoderConcat(nn.Module):
    "Implement the PE function with concatenation istead."
    "Output dimension will be (1,N,d_model*2)"
    def __init__(self, d_model, dropout = 0.0, max_seq_len = 1024):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Concatenate embeddings with positional sinusoid
        pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = torch.cat((x,pe),2)

        return self.dropout(x)

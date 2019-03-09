# Filename: Embedding.py
# Date Created: 08-Mar-2019 4:38:59 pm
# Author: zckoh
# Description: Embedding method before input to encoder.
import torch
import torch.nn as nn

class Embedder(nn.Module):
    """
        vocab_size = size of the dictionary of embeddings (88)
        d_model = size of each embedding vectors (4?)
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # TODO: Add positional sinusoid

    def forward(self, x):
        # TODO: Add forward pass
        return

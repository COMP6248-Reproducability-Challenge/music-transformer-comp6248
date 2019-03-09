# Filename: test_embedding.py
# Date Created: 09-Mar-2019 11:18:24 am
# Author: zckoh
# Description: file to use and test the embedding layers.
from OneHotEncoder import make_one_hot
from DataPrep import prepareData, tensorFromSequence
import numpy as np
import torch
from Embedding import Embedder, PositionalEncoder, PositionalEncoderConcat

# Prepare dataset from training split, generates pairs of input & target
pairs = prepareData('./datasets/JSB-Chorales-dataset/Jsb16thSeparated.npz',split='train')

# Use 1 pair as example
pair = pairs[0]

# Separate input & target
input = pair[0]
target = pair[1]

# Vocabulary size, 87 pitch values + 0 value for rest tokens (NaN in dataset).
vocab_size = 88
# Dimension of the model, arbitarily chosen as 4.
d_model = 4

# Initialize the layers.
embed = Embedder(vocab_size,d_model)
pos_encoder = PositionalEncoder(4)

# Test the implementation using the input tensors
# Typical flow through the embedding layer and positional encoder.
input = tensorFromSequence(input)
x = embed(input)
x = pos_encoder(x)

print(x.shape)

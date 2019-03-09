# Filename: Embedding.py
# Date Created: 08-Mar-2019 4:38:59 pm
# Author: zckoh
# Description: Embedding method before input to encoder.
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def make_one_hot(sequence, C=2):
    '''
    Represent the tokens in the sequence as onehot in pitch.

    Parameters
    ----------
    sequence : torch.Tensor
        N x 1, where N is the sequence length, after serializing the four voices.
        Each value is an integer representing the pitch.
    C : integer.
        number of pitch values.

    Returns
    -------
    target : torch.Tensor
        N x C, where C is pitch_values. One-hot encoded.
    '''
    one_hot = torch.Tensor(sequence.size(0), C).zero_()
    target = one_hot.scatter_(1, sequence.data, 1)

    target = Variable(target)

    return target

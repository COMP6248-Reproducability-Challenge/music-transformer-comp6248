# Filename: Process.py
# Date Created: 17-Mar-2019 5:43:02 pm
# Description: Functions used for basic processes.
import numpy as np
import copy

def IndexToPitch(input, vocab):
    """
    Converts the index values from model's output back to pitches from vocab.
    """
    index_vocab = np.arange(len(vocab))
    output = copy.deepcopy(input)

    for i, val in reversed(list(enumerate(index_vocab))):
        output[output==val] = vocab[i]

    return output

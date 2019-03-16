# Filename: DataPrep.py
# Date Created: 08-Mar-2019 10:01:18 pm
# Author: zckoh
# Description: Functions for preparing the dataset for training and evaluation.

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.autograd import Variable

def tensorFromSequence(sequence):
    """
    Generate tensors from the sequence in numpy.
    """
    output = Variable(torch.tensor(sequence)).long()
    return output


def PrepareData(npz_file, split='train', L=1024):
    """
    Function to prepare the data into pairs (input, target).
    Adds [PAD], [SOS] and [EOS] tokens into the data,
    where [PAD]=1, [SOS]=2, [EOS]=3.
    Limits the sequence to length of L.
    """
    print("Preparing data for",split,"split...")
    # Load in the data
    full_data = np.load(npz_file, fix_imports=True, encoding="latin1")
    data = full_data[split]

    # Initialize the tokens
    pad_token = np.array([[1]])
    sos_token = np.array([[2]])
    eos_token = np.array([[3]])

    # Repeat for all samples in data
    pairs = []
    for samples in data:
        # Serialise the dataset so that the resulting sequence is
        # S_1 A_1 T_1, B_2 S_2 A_2 T_2 B_2, ...

        # Generate input
        input_seq = samples.flatten()

        # Cut off the samples so that it has length of 1024
        if(len(input_seq) >= L):
            input_seq = input_seq[:L-1]

        # Set the NaN values to 0 and reshape accordingly
        input_seq = np.nan_to_num(input_seq.reshape(1,input_seq.size))
        # Add SOS token in the front
        input_seq = np.append(sos_token,input_seq, axis=1)

        # Generate target
        output_seq = input_seq[:,1:]
        # Add EOS token in the end
        output_seq = np.append(output_seq,eos_token, axis=1)

        # For both sequences, pad to sequence length L
        pad_array = pad_token * np.ones((1,1024-input_seq.shape[1]))
        input_seq = np.append(input_seq, pad_array,axis=1)
        pad_array = pad_token * np.ones((1,1024-output_seq.shape[1]))
        output_seq = np.append(output_seq, pad_array,axis=1)

        # Make it into a pair
        pair = [input_seq, output_seq]

        # Combine all pairs into one big list of pairs
        pairs.append(pair)

    print("Generated data pairs.")
    return pairs

def GenerateVocab(npz_file):
    """
    Generate vocabulary for the dataset including the custom tokens.
    """
    full_data = np.load(npz_file, fix_imports=True, encoding="latin1")
    train_data = full_data['train']
    validation_data = full_data['valid']
    test_data = full_data['test']

    combined_data = np.concatenate((train_data, validation_data, test_data))

    vocab = np.nan
    for sequences in combined_data:
        vocab = np.append(vocab,np.unique(sequences))

    vocab = np.unique(vocab)
    vocab = vocab[~np.isnan(vocab)]
    # Add rest, pad, sos and eos tokens.
    vocab = np.append([0,1,2,3],vocab)

    return vocab

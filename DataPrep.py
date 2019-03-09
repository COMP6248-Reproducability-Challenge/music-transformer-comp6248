# Filename: DataPrep.py
# Date Created: 08-Mar-2019 10:01:18 pm
# Author: zckoh
# Description: Functions for preparing the dataset for training and evaluation.

import torch
from torch.utils.data.dataset import Dataset
import numpy as np

# Generate tensors from the sequence to be used for training.
def tensorFromSequence(sequence):
    return torch.tensor(sequence)

# Function to prepare the data into pairs (input, target)
def prepareData(npz_file,split='train'):
    print("Preparing data for",split,"split...")
    full_data = np.load(npz_file, fix_imports=True, encoding="latin1")
    data = full_data[split]

    # Repeat for all samples in data
    pairs = []
    for samples in data:
        # Serialise the dataset so that the resulting sequence is
        # S_1 A_1 T_1, B_2 S_2 A_2 T_2 B_2, ...

        # Generate input
        input_seq = samples.flatten()
        # Set the NaN values to 0 and reshape accordingly
        input_seq = np.nan_to_num(input_seq.reshape(1,input_seq.size))

        # Generate target
        output_seq = input_seq[:,1:]

        # TODO: Append an EOS_token for the output

        # Make it into a pair
        pair = [input_seq, output_seq]

        # Combine all pairs into one big list of pairs
        pairs.append(pair)

    print("Generated data pairs.")
    return pairs

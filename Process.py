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

def ProcessModelOutput(model_output):
    """
    Remove custom tokens and set rest tokens to NaN values
    Converts the model's output into numpy format similar to JSB dataset.
    """
    # Values for the custom tokens
    rest_token = 0
    pad_token = 1
    sos_token = 2
    eos_token = 3

    # Convert tensor to numpy
    output = model_output.cpu().numpy()
    print("numpy shape:", output.shape)
    print(output[:30])

    # Remove SOS tokens
    output = output[output!=sos_token]

    # Remove EOS tokens
    output = output[output!=eos_token]

    # Remove pad tokens
    output = output[output!=pad_token]

    # Change rest tokens to NaN values
    output = np.where(output==rest_token, np.nan, output)

    # Reshape output to match JSB dataset
    output = output.reshape(round(output.shape[0]/4),4)

    return output

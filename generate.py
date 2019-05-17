import argparse
import torch
from DataPrep import GenerateVocab, PrepareData, tensorFromSequence
from Models import get_model
from MaskGen import create_masks
from Process import IndexToPitch, ProcessModelOutput, get_len
from Beam import beam_search
import numpy as np
import torch.nn.functional as F
import time
import os

def generate(model,opt):
    print("generating music using beam search...")
    model.eval()

    # choose 2 random pitches within the vocab (except rest/pad token) to start the sequence
    starting_pitch = torch.randint(2, len(opt.vocab)-1, (2,)).unsqueeze(1).transpose(0,1).to(opt.device)

    # generate the sequence using beam search
    generated_seq = beam_search(starting_pitch, model, opt)

    # Make the index values back to original pitch
    output_seq = IndexToPitch(generated_seq, opt.vocab)

    # Process the output format such that it is the same as our dataset
    processed = ProcessModelOutput(output_seq)

    return processed

def main():
    # Add parser to parse in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-load_weights', required=False)
    parser.add_argument('-output_name', type=str, required=True)
    parser.add_argument('-device', type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_ff', type=int, default=1024)
    parser.add_argument('-n_layers', type=int, default=5)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=1024)
    parser.add_argument('-attention_type', type = str, default = 'Baseline')
    parser.add_argument('-weights_name', type = str, default = 'model_weights')
    parser.add_argument("-concat_pos_sinusoid", type=str2bool, default=False)
    parser.add_argument("-relative_time_pitch", type=str2bool, default=False)
    parser.add_argument("-max_relative_position", type=int, default=512)
    opt = parser.parse_args()

    # Generate the vocabulary from the data
    opt.vocab = GenerateVocab(opt.src_data)
    opt.pad_token = 1

    # Create the model using the arguments and the vocab size
    model = get_model(opt, len(opt.vocab))

    # counter to keep track of how many outputs have been saved
    opt.save_counter = 0

    # Now lets generate some music
    generated_music = generate(model,opt)

    # Ask for next action
    promptNextAction(model, opt, generated_music)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, processed):
    while True:
        save = yesno(input('generate complete, save music? [y/n] : '))
        if save == 'y':
            print("saving music...")
            # Pickle the processed outputs for magenta later
            opt.save_counter += 1
            np.save('outputs/' + opt.output_name + str(opt.save_counter), processed)

        res = yesno(input("generate again? [y/n] : "))
        if res == 'y':
            # Now lets generate some music
            processed = generate(model,opt)

        else:
            print("exiting program...")
            break

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)
    main()

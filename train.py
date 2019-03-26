# Filename: train.py
# Date Created: 08-Mar-2019 11:48:49 pm
# Author: zckoh
# Description: Run this script to train the music-transformer model.
import argparse
import torch
from DataPrep import GenerateVocab, PrepareData, tensorFromSequence
from Models import get_model
from MaskGen import create_masks
from Process import IndexToPitch, ProcessModelOutput
import numpy as np

def train(model, opt):
    print("training model...")
    # TODO: Repeat for epochs

    # for i, pair in enumerate(opt.train):
    pair = opt.train[0]
    input = tensorFromSequence(pair[0]).to(opt.device)
    target = tensorFromSequence(pair[1]).to(opt.device)

    # Create mask for both input and target sequences
    input_mask, target_mask = create_masks(input, target, opt)

    preds_idx = model(input, target, input_mask, target_mask)
    # print(preds_idx.shape)

    # Make the index values back to original pitch
    preds = IndexToPitch(preds_idx, opt.vocab)
    print(preds.size(1))
    print(preds[0][:30])
    print(preds[0][-30:])

    # Process the preds format such that it is the same as our dataset
    processed = ProcessModelOutput(preds)
    print(processed.shape)

    # Pickle the processed outputs for magenta later
    np.save('outputs/train_output', processed)

def main():
    # Add parser to parse in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=False)
    parser.add_argument('-trg_data', required=False)
    parser.add_argument('-src_lang', required=False)
    parser.add_argument('-trg_lang', required=False)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_ff', type=int, default=1024)
    parser.add_argument('-n_layers', type=int, default=5)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=1024)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()

    # Set device to cuda if it is setup, else use cpu
    opt.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Generate the vocabulary from the data
    opt.vocab = GenerateVocab(opt.src_data)
    opt.pad_token = 1

    # Setup the dataset for training split
    opt.train = PrepareData(opt.src_data ,'train', int(opt.max_seq_len))

    # Create the model using the arguments and the vocab size
    model = get_model(opt, len(opt.vocab))

    # TODO: Set up optimizer for training

    # Train the model
    train(model, opt)

    # TODO: Evaluate the trained model


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    main()

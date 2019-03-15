# Filename: train.py
# Date Created: 08-Mar-2019 11:48:49 pm
# Author: zckoh
# Description: Run this script to train the music-transformer model.
import argparse
import torch

def train():
    raise NotImplementedError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=False)
    parser.add_argument('-trg_data', required=False)
    parser.add_argument('-src_lang', required=False)
    parser.add_argument('-trg_lang', required=False)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()

if __name__ == "__main__":
    main()

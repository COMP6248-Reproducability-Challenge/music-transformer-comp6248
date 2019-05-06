import argparse
import torch
from DataPrep import GenerateVocab, PrepareData, tensorFromSequence
from Models import get_model
from MaskGen import create_masks
from Process import IndexToPitch, ProcessModelOutput, get_len
import numpy as np
import torch.nn.functional as F
import time
import os

def generate(model,opt):

    model.eval()
    pitch = []

    #choose a random pitch within pitch index (2 to ) to start the sequence
    starting_pitch = torch.randint(2, len(opt.vocab)-1, (1,))

    pitch = beam_search(starting_pitch, model, opt)




def main():
    # Add parser to parse in the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-src_data', required=True)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_ff', type=int, default=1024)
    parser.add_argument('-n_layers', type=int, default=5)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=1024)

    parser.add_argument('-load_weights', required=True)

    parser.add_argument('-attention_type', type = str, default = 'Baseline')
    parser.add_argument('-weights_name', type = str, default = 'model_weights')
    parser.add_argument("-concat_pos_sinusoid", type=str2bool, default=False)
    parser.add_argument("-relative_time_pitch", type=str2bool, default=False)
    opt = parser.parse_args()

    # Generate the vocabulary from the data
    opt.vocab = GenerateVocab(opt.src_data)
    opt.pad_token = 1

    # Create the model using the arguments and the vocab size
    model = get_model(opt, len(opt.vocab))


    if opt.floyd is False:
        promptNextAction(model, opt, epoch, step_num, avg_loss)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, epoch, step_num, avg_loss):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0

    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            print("saving weights...")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.optimizer.state_dict(),
            'loss': avg_loss,
            'step_num': step_num
            }, 'weights/' + opt.weights_name)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                #epochs = input("type number of epochs to train for : ")
                try:
                #    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
        #    opt.epochs = epochs
        #    opt.resume = True

            train(model, opt)
        else:
            print("exiting program...")
            break


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    main()

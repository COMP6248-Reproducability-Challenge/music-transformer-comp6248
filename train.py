# Filename: train.py
# Date Created: 08-Mar-2019 11:48:49 pm
# Author: zckoh
# Description: Run this script to train the music-transformer model.
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

def train(model, opt):
    print("training model...")
    # TODO: Repeat for epochs
    model.train()
    start = time.time()
    warmup_steps = 4000
    step_num_load = 0

    if opt.checkpoint > 0:
        cptime = time.time()

    # if load_weights to resume training
    if opt.load_weights is not None:
        checkpoint = torch.load('weights/' + opt.weights_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_num_load = checkpoint['step_num']  # to keep track of learning rate

    if opt.resume is True:
        checkpoint = torch.load('weights/' + opt.weights_name)

        # No need to load weights, as it is the same model being trained

        # model.load_state_dict(checkpoint['model_state_dict'])
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_num_load = checkpoint['step_num']  # to keep track of learning rate


    for epoch in range(opt.epochs):

        # step_num_load == loaded step_num to pick up from last learning rate
        step_num = step_num_load + time.time() - start

        # learning rate defined in Attention is All You Need Paper
        opt.lr = (opt.d_model ** (-0.5)) * (min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5)))


        # Vary learning rate based on step numbers (each note consider a step)
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)


        total_loss = []
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')

        for i, batch in enumerate(opt.train):

            pair = batch
            input = tensorFromSequence(pair[0]).to(opt.device)
            target = tensorFromSequence(pair[1]).to(opt.device)
            # trg_input = target[:,:-1]
            trg_input = target
            ys = target[:, 0:].contiguous().view(-1)

            # Create mask for both input and target sequences
            input_mask, target_mask = create_masks(input, trg_input, opt)

            preds_idx = model(input, trg_input, input_mask, target_mask)



            opt.optimizer.zero_grad()
    #         loss = torch.nn.CrossEntropyLoss(ignore_index = opt.pad_token)
            loss = F.cross_entropy(preds_idx.contiguous().view(preds_idx.size(-1), -1).transpose(0,1), ys, \
                                   ignore_index = opt.pad_token)
    #         loss(preds_idx.contiguous().view(preds_idx.size(-1), -1).transpose(0,1), ys)
            loss.backward()
            opt.optimizer.step()

    #         if opt.SGDR == True:
    #             opt.sched.step()

    #         total_loss += loss.item()

            total_loss.append(loss.item())

            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / get_len(opt.train))
                avg_loss = np.mean(total_loss)
                if opt.floyd is False:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end ='\r')
                else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
    #             total_loss = 0

            # checkpoint in terms of minutes reached, then save weights, and other information
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                print("checkpoint save...")
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.optimizer.state_dict(),
                'loss': avg_loss,
                'step_num': step_num
                }, 'weights/' + opt.weights_name)
                cptime = time.time()

        avg_loss = np.mean(total_loss)

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

    return epoch, avg_loss, step_num

    #         # Make the index values back to original pitch
    #         preds = IndexToPitch(preds_idx, opt.vocab)
    #         print(preds.size(1))
    #         print(preds[0][:30])
    #         print(preds[0][-30:])

    #         # Process the preds format such that it is the same as our dataset
    #         processed = ProcessModelOutput(preds)
    #         print(processed.shape)

    #         # Pickle the processed outputs for magenta later
    #         np.save('outputs/train_output', processed)

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
    parser.add_argument('-dropout', type=int, default=0.0)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=1024)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type= float, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-attention_type', type = str, default = 'Baseline')
    parser.add_argument('-weights_name', type = str, default = 'model_weights')

    opt = parser.parse_args()

    # Initialize resume option as False
    opt.resume = False

    # Set device to cuda if it is setup, else use cpu
    opt.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    # Generate the vocabulary from the data
    opt.vocab = GenerateVocab(opt.src_data)
    opt.pad_token = 1


    # Setup the dataset for training split
    opt.train = PrepareData(opt.src_data ,'train', int(opt.max_seq_len))

    # Create the model using the arguments and the vocab size
    model = get_model(opt, len(opt.vocab))

    # TODO: Set up optimizer for training
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    # Train the model
    step_num = 0 # step_num is based on time, which is used to calculate learning rate

    avg_loss, epoch, step_num = train(model, opt)

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
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            opt.resume = True

            train(model, opt)
        else:
            print("exiting program...")
            break

    # TODO: Evaluate the trained model


if __name__ == "__main__":
    # For reproducibility
    torch.manual_seed(0)

    main()

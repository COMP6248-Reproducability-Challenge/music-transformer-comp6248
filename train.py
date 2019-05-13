# Filename: train.py
# Date Created: 08-Mar-2019 11:48:49 pm
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

torch.set_default_dtype(torch.float)
torch.set_default_tensor_type(torch.FloatTensor)

def count_nonpad_tokens(target, padding_index):
    nonpads = (target != padding_index).squeeze()
    ntokens = torch.sum(nonpads)
    return ntokens


def LabelSmoothing(input, target, smoothing, padding_index):
    """
    Args:
        input: input to loss function, size of [N, Class]
        target: target input to loss function
        smoothing: degree of smoothing
        padding_index: number used for padding, i.e. 1

    Returns:
        A smoothed target input to loss function
    """
    confidence = 1.0 - smoothing
    true_dist = input.clone()
    true_dist.fill_(smoothing/ (input.size(1) - 2))
    true_dist.scatter_(1, target.unsqueeze(1), confidence)
    mask = torch.nonzero(target == padding_index)
    if mask.dim() > 0:
        true_dist.index_fill_(0, mask.squeeze(), 0.0)

    return torch.autograd.Variable(true_dist, requires_grad = False)

def train(model, opt):
    print("training model...")
    start = time.time()
    warmup_steps = 4000
    step_num_load = 0
    step_num = 1
    epoch_load = 0

    t_loss_per_epoch = []
    v_loss_per_epoch = []

    if opt.checkpoint > 0:
        cptime = time.time()

    # if load_weights to resume training
    if opt.load_weights is not None:
        checkpoint = torch.load('weights/' + opt.weights_name, map_location = 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_num_load = checkpoint['step_num']  # to keep track of learning rate
        epoch_load = checkpoint['epoch']

    if opt.resume is True:
        checkpoint = torch.load('weights/' + opt.weights_name)

        # No need to load weights, as it is the same model being trained
        # model.load_state_dict(checkpoint['model_state_dict'])

        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_num_load = checkpoint['step_num']  # to keep track of learning rate
        epoch_load = checkpoint['epoch']
    step_num += step_num_load
    epoch_load += epoch_load

    for epoch in range(opt.epochs):
        model.train()

        # learning rate defined in Attention is All You Need Paper
        opt.lr = (opt.d_model ** (-0.5)) * (min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5)))

        # Vary learning rate based on step numbers (each note consider a step)
        for param_group in opt.optimizer.param_groups:
            param_group['lr'] = opt.lr

        total_loss = []
        print("   %dm: epoch %d [%s]  %d%%  training loss = %s" %\
        ((time.time() - start)//60, (epoch + epoch_load) + 1, "".join(' '*20), 0, '...'), end='\r')

        for i, batch in enumerate(opt.train):
            pair = batch
            input = tensorFromSequence(pair[0]).to(opt.device)
            target = tensorFromSequence(pair[1]).to(opt.device)
            # input = pair[0].to(opt.device)
            # target = pair[1].to(opt.device)

            # trg_input = target[:,:-1]
            trg_input = target
            ys = target[:, 0:].contiguous().view(-1)

            # Create mask for both input and target sequences
            input_mask, target_mask = create_masks(input, trg_input, opt)

            preds_idx = model(input, trg_input, input_mask, target_mask)

            opt.optimizer.zero_grad()

            loss_input = preds_idx.contiguous().view(preds_idx.size(-1), -1).transpose(0,1)
            smoothed_target = LabelSmoothing(loss_input, ys, 0.1, 1)
            criterion =  torch.nn.KLDivLoss(size_average = False)
            # loss = criterion(loss_input, smoothed_target)/ (count_nonpad_tokens(ys, 1))
            # print(count_nonpad_tokens(ys, 1))
            # loss = F.binary_cross_entropy_with_logits(loss_input, smoothed_target, size_average = False) / (count_nonpad_tokens(ys, 1))
            # loss = F.cross_entropy(preds_idx.contiguous().view(preds_idx.size(-1), -1).transpose(0,1), ys, \
            #                        ignore_index = opt.pad_token, size_average = False) / (count_nonpad_tokens(ys,1))
            loss = F.nll_loss(F.log_softmax(loss_input,dim = 1), ys, ignore_index = 1)
            # print(F.softmax(loss_input,dim = 1))
            loss.backward()
            opt.optimizer.step()
            step_num += 1

            total_loss.append(loss.item())

            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / get_len(opt.train))
                avg_loss = np.mean(total_loss)

                print("   %dm: epoch %d [%s%s]  %d%%  training loss = %.3f" %\
                ((time.time() - start)//60, (epoch + epoch_load) + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end ='\r')

            # checkpoint in terms of minutes reached, then save weights, and other information
            if opt.checkpoint > 0 and ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                print("checkpoint save...")
                torch.save({
                'epoch': epoch + epoch_load,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.optimizer.state_dict(),
                'loss': avg_loss,
                'step_num': step_num
                }, 'weights/' + opt.weights_name)
                cptime = time.time()

        avg_loss = np.mean(total_loss)

        # Validation step
        model.eval()
        total_validate_loss = []
        with torch.no_grad():
            for i, batch in enumerate(opt.valid):
                pair = batch
                input = tensorFromSequence(pair[0]).to(opt.device)
                target = tensorFromSequence(pair[1]).to(opt.device)
                trg_input = target
                ys = target[:, 0:].contiguous().view(-1)

                input_mask, target_mask = create_masks(input, trg_input, opt)
                preds_validate = model(input, trg_input, input_mask, target_mask)

                criterion = torch.nn.KLDivLoss(size_average = False)
                validate_input = preds_validate.contiguous().view(preds_validate.size(-1), -1).transpose(0,1)
                smoothed_target_validate = LabelSmoothing(validate_input, ys, 0.0, 1)
                validate_loss = criterion(validate_input, smoothed_target_validate)/ (count_nonpad_tokens(ys, 1))
                # validate_loss = F.cross_entropy(preds_validate.contiguous().view(preds_validate.size(-1), -1).transpose(0,1), ys, \
                #                        ignore_index = opt.pad_token, size_average = False) / (count_nonpad_tokens(ys, 1))
                validate_loss = F.nll_loss(F.log_softmax(validate_input, dim = 1), ys, ignore_index = 1)
                # validate_loss = F.binary_cross_entropy_with_logits(validate_input, smoothed_target_validate, size_average = False) / (count_nonpad_tokens(ys, 1))
                total_validate_loss.append(validate_loss.item())
            avg_validate_loss = np.mean(total_validate_loss)

        # Store the average training & validation loss for each epoch
        t_loss_per_epoch.append(avg_loss)
        v_loss_per_epoch.append(avg_validate_loss)

        print("%dm: epoch %d [%s%s]  %d%%  training loss = %.3f\nepoch %d complete, training loss = %.03f, validation loss = %.03f" %\
        ((time.time() - start)//60, (epoch + epoch_load) + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, (epoch + epoch_load) + 1, avg_loss, avg_validate_loss))

    return epoch, avg_loss, step_num, t_loss_per_epoch, v_loss_per_epoch

def main():
    # Add parser to parse in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=False)
    parser.add_argument('-device', type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-epochs', type=int, default=1)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-d_ff', type=int, default=1024)
    parser.add_argument('-n_layers', type=int, default=5)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-max_seq_len', type=int, default=1024)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type= float, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-attention_type', type = str, default = 'Baseline')
    parser.add_argument('-weights_name', type = str, default = 'model_weights')
    parser.add_argument("-concat_pos_sinusoid", type=str2bool, default=False)
    parser.add_argument("-relative_time_pitch", type=str2bool, default=False)
    parser.add_argument("-max_relative_position", type= int, default = 512)
    opt = parser.parse_args()

    # Initialize resume option as False
    opt.resume = False

    # Generate the vocabulary from the data
    opt.vocab = GenerateVocab(opt.src_data)
    opt.pad_token = 1

    # Setup the dataset for training split and validation split
    opt.train = PrepareData(opt.src_data ,'train', int(opt.max_seq_len))
    opt.valid = PrepareData(opt.src_data ,'valid', int(opt.max_seq_len))

    # Create the model using the arguments and the vocab size
    model = get_model(opt, len(opt.vocab))

    # Set up optimizer for training
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)

    # step_num is based on time, which is used to calculate learning rate
    step_num = 0

    # Train the model
    avg_loss, epoch, step_num, t_loss_per_epoch, v_loss_per_epoch = train(model, opt)

    promptNextAction(model, opt, epoch, step_num, avg_loss, t_loss_per_epoch, v_loss_per_epoch)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, epoch, step_num, avg_loss, t_loss_per_epoch, v_loss_per_epoch):

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

            _, _, _, extra_t_loss, extra_v_loss = train(model, opt)
            t_loss_per_epoch.extend(extra_t_loss)
            v_loss_per_epoch.extend(extra_v_loss)
        else:
            # Convert list into numpy arrays
            t_loss_per_epoch = np.array(t_loss_per_epoch)
            v_loss_per_epoch = np.array(v_loss_per_epoch)

            # Save the arrays for plotting later
            np.save('outputs/t_loss%d'%int(time.time()//60), t_loss_per_epoch)
            np.save('outputs/v_loss%d'%int(time.time()//60), v_loss_per_epoch)

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

# Filename: Beam.py
# Date Created: 15-Mar-2019 2:42:12 pm
# Description: Functions used for beam search.
import torch
import torch.nn.functional as F
from MaskGen import nopeak_mask
import math

def init_vars(src, model, opt):
    # outputs used for the decoder is the starting pitches
    outputs = src

    # encoder pass
    src_mask = (src != opt.pad_token).unsqueeze(-2).to(opt.device)
    e_output = model.encoder(src, src_mask)

    # decoder pass
    trg_mask = nopeak_mask(src.shape[1], opt)
    out = model.decoder(outputs, e_output, src_mask, trg_mask)
    out = model.linear(out)

    # final fc layer
    out = F.softmax(out, dim=-1)

    # calculate probablites for beam search
    # takes the last output from the model, hence out[:, -1]
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    # store the model outputs
    outputs = torch.zeros(opt.k, opt.max_seq_len).long().to(opt.device)
    outputs[:, 0:src.shape[1]] = src
    outputs[:, src.shape[1]] = ix[0]

    # store the encoder output to be used later
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1)).to(opt.device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    # calculate probablities for each step in the sequence
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    # update outputs
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores

def beam_search(src, model, opt):
    outputs, e_outputs, log_scores = init_vars(src, model, opt)
    init_start_len = outputs.shape[0]
    src_mask = (src != opt.pad_token).unsqueeze(-2).to(opt.device)

    for i in range(init_start_len, opt.max_seq_len):
        # Just comment this block of code if only use encoder once at the start
        src_mask = (outputs[0,:i].unsqueeze(-2) != opt.pad_token).unsqueeze(-2).to(opt.device)
        e_output = model.encoder(outputs[0,:i].unsqueeze(-2), src_mask)
        e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1)).to(opt.device)
        e_outputs[:, :] = e_output[0]

        trg_mask = nopeak_mask(i, opt)
        out = model.linear(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)

    # return the one with the largest log_scores
    return outputs[0]

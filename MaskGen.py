# Filename: MaskGen.py
# Date Created: 15-Mar-2019 2:42:12 pm
# Description: Functions used to generate masks w.r.t. given inputs.
import torch
import numpy as np
from torch.autograd import Variable

def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0).to(opt.device)
    return np_mask

def create_masks(src, trg, opt):
    src_mask = (src != opt.pad_token).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.pad_token).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask

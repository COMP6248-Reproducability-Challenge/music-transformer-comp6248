import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def attention(q, v, k, d_k, mask = None, dropout = None):

    scores = torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1) #shape of mask must be broadcastable with shape of underlying tensor
        scores = scores.masked_fill(mask == 0, -1e9) #masked_fill fills elements of scores with -1e9 where mask == 0

    scores = F.softmax(scores, dim = -1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)

    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  #final dimension = d_model/N as we split embedding vec into N heads
        self.h = heads #number of heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0) #batch size

        # perform linear operation and split into h heads

        #q = torch.zeros(512,3,512)
        # x = nn.Linear(512,512)
        # x = x(q).view(512, -1, 8, 512//8) will result in torch.Size([512, 3, 8, 64])

        #original size bs * seq_len * h * d_k
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions of bs * h * seq_len * d_k

        k = k.transpose(1,2) # torch.Size([512, 3, 8, 64]) transpose will result in torch.Size([512, 8, 3, 64])
        q = q.transpose(1,2)
        v = v.transpose(1,2)

    # calculate attention using defined attention function
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        #concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        #create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.ones(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim = 1, keepdim = True)) \
        / (x.std(dim = 1, keepdim = True) + self.eps) + self.bias

        return norm

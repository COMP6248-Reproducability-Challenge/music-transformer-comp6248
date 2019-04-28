import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def shape_list(x):
    """Return list of dims
    """

    shape = list(x.shape)

    return shape


def _relative_position_to_absolute_position_masked(x):
    """Helper function for dot_product_self_attention_relative

    Rearrange attention logits or weights tensor.

    Dimensions of input represents:
    [batch, heads, query_position, memory_position - query_position + length - 1]

    Dimensions of output represents:
    [batch, heads, query_position, memory_position]

    Only works with masked attention.

    Args:
        x: a Tensor with shape [batch, heads, length, length]

    Returns:
        a Tensor with shape [batch, heads, length, length]
    """

    batch, heads, length, _ = shape_list(x)

    x = F.pad(x, (1, 0, 0, 0, 0, 0, 0, 0))
    # print("After pad: ", x.shape)
    x = torch.reshape(x, (batch, heads, 1 + length, length))
    # print(x.shape)
    x = x[0:x.shape[0] - 0, 0:x.shape[1] - 0, 1:x.shape[2], 0:x.shape[3] - 0]

    return x


def matmul_with_relative_keys(x, y, heads_share_relative_embedding):
    if heads_share_relative_embedding:
        ret = torch.einsum("bhld,md -> bhlm", x, y)
    else:
        ret = torch.einsum("bhld,hmd -> bhlm", x, y)
    return ret


def get_relative_embeddings_left(max_relative_position, length, depth,
                                num_heads,
                                heads_share_relative_embedding):
    """Instantiate or retrieve relative embeddings, sliced according to length

    Use for masked case where the relative attention is only looking left
    Args:
        max_relative_position: an Integer for the number of entries in the relative
          embedding, which corresponds to the max relative distance that is
          considered.
        length: an Integer, specifies the length of the input sequence for which
          this relative embedding is retrieved for.
        depth: an Integer, specifies the depth for relative embeddings.
        num_heads: an Integer, specifies the number of heads.
        heads_share_relative_embedding: a Boolean specifying if the relative
          embedding is shared across heads.
    """

    initializer_stddev = depth ** -0.5
    if heads_share_relative_embedding:
        embedding_shape = (max_relative_position, depth)
    else:
        embedding_shape = (num_heads, max_relative_position, depth)

    relative_embeddings = Variable(torch.from_numpy(np.random.normal(0.0, initializer_stddev, embedding_shape).astype('f')))
    pad_length = max(length - max_relative_position, 0)
    slice_start_position = max(max_relative_position - length, 0)

    if heads_share_relative_embedding:
        padded_relative_embeddings = F.pad(
            relative_embeddings,
            (0, 0, pad_length, 0))
    # used_relative_embeddings = tf.slice(
    #     padded_relative_embedding,
    #     [start_slice_position, 0], [length, -1])
        used_relative_embeddings = padded_relative_embeddings[slice_start_position:length,
                                                0:(padded_relative_embeddings.shape[1] - 0)]
    else:
    # padded_relative_embeddings = tf.pad(
    #     relative_embeddings,
    #     [[0, 0], [pad_length, 0], [0, 0]])
    # used_relative_embeddings = tf.slice(
    #     padded_relative_embeddings,
    #     [0, start_slice_position, 0], [-1, length, -1])
        # padded_relative_embeddings = F.pad(
        #     relative_embeddings,
        #     (0, 0, pad_length, 0, 0, 0))
        padded_relative_embeddings = F.pad(
            relative_embeddings,
            (0, 0, pad_length, 0, 0, 0))
        used_relative_embeddings = padded_relative_embeddings[
                                    0:(padded_relative_embeddings.shape[0] - 0),
                                    slice_start_position:length,
                                    0:(padded_relative_embeddings.shape[2] - 0)
                                    ]

    return used_relative_embeddings


def dot_product_self_attention_relative(q,
                                        k,
                                        v,
                                        mask = None,
                                        bias = None,
                                        max_relative_position = None,
                                        dropout = None,
                                        heads_share_relative_embedding = False):
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                     "relative self attention." % (max_relative_position))

    # Use separate embeddings suitable for keys and values.
    _, heads, length, depth_k = shape_list(k)

    logits = torch.matmul(q, k.transpose(-2, -1))

    # print("q is :     ", q.shape)

    if mask is not None:
        mask = mask.unsqueeze(1) #shape of mask must be broadcastable with shape of underlying tensor
        logits = logits.masked_fill(mask == 0, -1e9) #masked_fill fills elements of scores with -1e9 where mask == 0

    # print("logits:     ", logits.shape)

    key_relative_embeddings = get_relative_embeddings_left(
        max_relative_position, length, depth_k, heads, heads_share_relative_embedding).to(q.device)
    relative_logits = matmul_with_relative_keys(q, key_relative_embeddings,
                                                heads_share_relative_embedding)

    relative_logits = _relative_position_to_absolute_position_masked(relative_logits)  #[1, 8, 1023, 1024]
    logits += relative_logits

    if bias is not None:
        logits += bias

    weights = F.softmax(logits, dim = -1)
    # Dropping out the attention links for each of the heads.
    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, v)

    return output


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
    def __init__(self, heads, d_model, dropout = 0.0, attention_type = "Baseline",
                                                        bias = None,
                                                        max_relative_position = 512,
                                                        heads_share_relative_embedding = False):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads  #final dimension = d_model/N as we split embedding vec into N heads
        self.h = heads #number of heads

        self.attention_type = attention_type
        self.bias = bias
        self.max_relative_position = max_relative_position
        self.heads_share_relative_embedding = heads_share_relative_embedding

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
        if self.attention_type == "Baseline":
            scores = attention(q, k, v, self.d_k, mask, self.dropout)

        elif self.attention_type == "dot_product_self_attention_relative":
            scores = dot_product_self_attention_relative(q, k, v, mask,
                                                                    self.bias,
                                                                    self.max_relative_position,
                                                                    self.dropout,
                                                                    self.heads_share_relative_embedding)

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

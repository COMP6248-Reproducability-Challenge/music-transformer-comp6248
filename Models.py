# Filename: Models.py
# Date Created: 16-Mar-2019 2:17:09 pm
# Description: Combine all sublayers into one tranformer model.
import torch
import torch.nn as nn
from Encode_Decode_Layers import EncoderLayer, DecoderLayer
from Embedding import Embedder, PositionalEncoder
from Sublayers import Norm
import torch.nn.functional as F
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_len, d_ff):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout, max_seq_len)
        self.layers = get_clones(EncoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_len, d_ff):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout, max_seq_len)
        self.layers = get_clones(DecoderLayer(d_model, heads, d_ff, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, heads,
                 dropout, max_seq_len, d_ff):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, N, heads, dropout, max_seq_len, d_ff)
        self.decoder = Decoder(trg_vocab_size, d_model, N, heads, dropout, max_seq_len, d_ff)
        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        linear_output = self.linear(d_output)

        # Add log softmax layer as the top layer
        log_softmax_outputs = F.log_softmax(linear_output,dim=2)

        # Find the small value and take as the output
        outputs = log_softmax_outputs.argmin(dim=2)
        return outputs

def get_model(opt, vocab_size):
    # Ensure the provided arguments are valid
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    # Initailze the transformer model
    model = Transformer(vocab_size, vocab_size, opt.d_model, opt.n_layers,
                        opt.heads, opt.dropout, opt.max_seq_len, opt.d_ff)

    # if opt.load_weights is not None:
    #     print("loading pretrained weights...")
    #     model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    # else:
    #     for p in model.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #
    model = model.to(opt.device)

    return model

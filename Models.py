# Filename: Models.py
# Date Created: 16-Mar-2019 2:17:09 pm
# Description: Combine all sublayers into one tranformer model.
import torch
import torch.nn as nn
from Encode_Decode_Layers import EncoderLayer, DecoderLayer
from Embedding import Embedder, PositionalEncoder, PositionalEncoderConcat
from Sublayers import Norm
import torch.nn.functional as F
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, opt):
        super().__init__()
        self.N = opt.n_layers
        self.embed = Embedder(vocab_size, opt.d_model)
        if opt.concat_pos_sinusoid is True:
            self.pe = PositionalEncoderConcat(opt.d_model, opt.dropout, opt.max_seq_len)
            self.d_model = 2 * opt.d_model
        else:
            self.pe = PositionalEncoder(opt.d_model, opt.dropout, opt.max_seq_len)
            self.d_model = opt.d_model

        if opt.relative_time_pitch is True:
            self.layers = get_clones(EncoderLayer(self.d_model, opt.heads, opt.d_ff, \
                                                opt.dropout, opt.attention_type, \
                                                opt.relative_time_pitch,
                                                max_relative_position = opt.max_relative_position),
                                                opt.n_layers)
            self.layers.insert(0, copy.deepcopy(EncoderLayer(self.d_model, opt.heads, opt.d_ff, \
                                                opt.dropout, opt.attention_type, \
                                                relative_time_pitch = False,
                                                max_relative_position = opt.max_relative_position)))
        else:
            self.layers = get_clones(EncoderLayer(self.d_model, opt.heads, opt.d_ff, \
                                                opt.dropout, opt.attention_type, \
                                                opt.relative_time_pitch,
                                                max_relative_position = opt.max_relative_position),
                                                opt.n_layers)
        self.norm = Norm(self.d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        for i in range(self.N):
            x = self.layers[i](x.float(), mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, opt):
        super().__init__()
        self.N = opt.n_layers
        self.embed = Embedder(vocab_size, opt.d_model)
        if opt.concat_pos_sinusoid is True:
            self.pe = PositionalEncoderConcat(opt.d_model, opt.dropout, opt.max_seq_len)
            self.d_model = 2 * opt.d_model
        else:
            self.pe = PositionalEncoder(opt.d_model, opt.dropout, opt.max_seq_len)
            self.d_model = opt.d_model

        if opt.relative_time_pitch is True:
            self.layers = get_clones(DecoderLayer(self.d_model, opt.heads, opt.d_ff, \
                                                opt.dropout, opt.attention_type, \
                                                opt.relative_time_pitch,
                                                max_relative_position = opt.max_relative_position),
                                                opt.n_layers-1)
            self.layers.insert(0, copy.deepcopy(DecoderLayer(self.d_model, opt.heads, opt.d_ff, \
                                                opt.dropout, opt.attention_type, \
                                                relative_time_pitch = False,
                                                max_relative_position = opt.max_relative_position)))

        else:
            self.layers = get_clones(DecoderLayer(self.d_model, opt.heads, opt.d_ff, \
                                                opt.dropout, opt.attention_type, \
                                                opt.relative_time_pitch,
                                                max_relative_position = opt.max_relative_position),
                                                opt.n_layers)
        self.norm = Norm(self.d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        # print(x.shape)
        for i in range(self.N):
            x = self.layers[i](x.float(), e_outputs, src_mask, trg_mask)

        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, opt):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, opt)
        self.decoder = Decoder(trg_vocab_size, opt)
        if opt.concat_pos_sinusoid is True:
            self.d_model = 2 * opt.d_model
        else:
            self.d_model = opt.d_model

        self.linear = nn.Linear(self.d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):

        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.linear(d_output)

        return output

def get_model(opt, vocab_size):
    # Ensure the provided arguments are valid
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    print('Attention type: ' + opt.attention_type)

    # Initailze the transformer model
    model = Transformer(vocab_size, vocab_size, opt)

    if opt.load_weights is not None:
        print("loading pretrained weights...")
        checkpoint = torch.load(f'{opt.load_weights}/' + opt.weights_name, map_location = 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(opt.device)

    return model

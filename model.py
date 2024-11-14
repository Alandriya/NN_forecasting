from torch import nn
from config import cfg
import torch
import numpy as np
import math
from collections import OrderedDict


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4], dilation=v[5])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = cfg.CONV_conv(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4], dilation=v[5])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
        elif 'transformer_encoder' in layer_name:
            encoder_layer = nn.TransformerEncoderLayer(d_model=v[0], nhead=8)
            layers.append((layer_name, nn.TransformerEncoder(encoder_layer, num_layers=6)))
        elif 'transformer_decoder' in layer_name:
            decoder_layer = nn.TransformerDecoderLayer(d_model=v[0], nhead=8)
            layers.append((layer_name, nn.TransformerDecoder(decoder_layer, num_layers=6)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Model(nn.Module):
    def __init__(self, encoder, rnn, decoder):
        super().__init__()
        self.encoder = make_layers(encoder)
        self.rnns = rnn
        self.decoder = make_layers(decoder)

    def forward(self, inputs, mode=''):
        x, eta, epoch = inputs  # s b c h w

        layer_hiddens = None
        m = None

        x = torch.nn.functional.pad(x, (2, 3, 7, 8))

        # b c in h w
        input = x[:, :cfg.in_len]
        # print(input.shape) torch.Size([8, 7, 3, 81, 91])

        # b in c h w -> b in*c h w
        input = torch.reshape(input, (input.shape[0], input.shape[1] * input.shape[2], input.shape[3], input.shape[4]))
        output, m, layer_hiddens, decouple_loss = self.rnns(input, m, layer_hiddens, self.encoder, self.decoder)

        # print(output.shape)
        # b out*c h w -> b out c h w
        output = torch.reshape(output, (-1, cfg.out_len, 3, output.shape[2], output.shape[3]))
        # print(output.shape)

        output = output[:, :, :, 7:81 + 7, 2:91 + 2]
        return output, decouple_loss

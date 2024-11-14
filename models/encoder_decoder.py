import torch
import torch.nn as nn
from config import cfg


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x


class Encoder_Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = b_h_w[0]
        self.height = b_h_w[1]
        self.width = b_h_w[2]
        self._Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._Conv1 = conv_block(ch_in=input_channel, ch_out=64)
        self._Conv2 = conv_block(ch_in=64, ch_out=128)

        self._Up = up_conv(ch_in=128, ch_out=64)
        self._Conv_1x1 = nn.Conv2d(64, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def encode(self, x):
        # print(x.shape) # torch.Size([16, 20, 6, 81, 91])
        x = torch.nn.functional.pad(x, (2, 3, 7, 8))
        # print(x.shape) # torch.Size([16, 20, 6, 96, 96])
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        # print(x.shape) #torch.Size([16, 120, 96, 96])
        x1 = self._Conv1(x)
        x2 = self._Maxpool(x1)
        x2 = self._Conv2(x2)
        return x2

    def decode(self, y):
        y1 = self._Up(y)
        # print(y1.shape) # torch.Size([16, 64, 96, 96])
        y2 = self._Conv_1x1(y1)
        # print(y2.shape) # torch.Size([16, 60, 94, 94])
        output = torch.reshape(y2, (-1, cfg.out_len, cfg.features_amount, y2.shape[2], y2.shape[3]))
        # print(output.shape) # torch.Size([16, 10, 3, 94, 94])
        output = output[:, :, :, 7:self.height + 7, 2:self.width + 2]
        # print(output.shape) #torch.Size([16, 10, 3, 81, 91])

        return output

    def forward(self, x, mode):
        if mode == 'train':
            y = self.encode(x)
            # print(y.shape)
            return self.decode(y)
        elif mode == 'encode':
            return self.encode(x)
        elif mode == 'decode':
            return self.decode(x)
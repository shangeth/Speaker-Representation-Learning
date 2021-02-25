import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import numpy as np
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Encoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.drop_prob = 0.1
        self.channels = channels
        self.encoder = nn.Sequential(
            # SincConv_fast(self.channels, 251), 
            nn.Conv1d(1, self.channels, 10, 5),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 8, 4),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 4, 2),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 4, 2),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 4, 2),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 1, 1),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class SincConv_fast(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes


    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)
        
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        band_pass = band_pass / (2*band[:,None])
        
        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 



class Attention(nn.Module):
    def __init__(self, attn_dim):
        super().__init__()
        self.attn_dim = attn_dim
        self.W = nn.Parameter(torch.Tensor(self.attn_dim, self.attn_dim), requires_grad=True)
        self.v = nn.Parameter(torch.Tensor(1, self.attn_dim), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.attn_dim)
        for weight in self.W:
            nn.init.uniform_(weight, -stdv, stdv)
        for weight in self.v:
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, inputs, attn=False):
        # inputs = inputs.transpose(1,2)
        batch_size = inputs.size(0)
        weights = torch.bmm(self.W.unsqueeze(0).repeat(batch_size, 1, 1), inputs)
        e = torch.tanh(weights)
        e = torch.bmm(self.v.unsqueeze(0).repeat(batch_size, 1, 1), e)
        attentions = torch.softmax(e.squeeze(1), dim=-1)
        weighted = torch.mul(inputs, attentions.unsqueeze(1).expand_as(inputs))
        representations = weighted.sum(2).squeeze()
        if attn:
            return representations, attentions
        else:
            return representations

class Discriminator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        # self.encoder = nn.Sequential()
        # self.a1 = Attention(dim)
        # self.a2 = Attention(dim)

        self.classifier = nn.Sequential(
            nn.Linear(int(self.dim),  int(self.dim/2)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(int(self.dim/2), 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        # print(z1.shape)
        # z1 = self.a1(z1)
        # z2 = self.a2(z2)
        y_hat = self.classifier(torch.cat([z1, z2], 1))
        # y_hat = self.classifier(torch.cat([z1.transpose(1,2).mean(1), z2.transpose(1,2).mean(1)], 1))
        return y_hat

class Accumulator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=8)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # self.lstm = nn.LSTM(dim, int(self.dim/2), batch_first=True)
        # self.a = Attention(int(self.dim/4))

    def forward(self, z):
        # print(z.shape)
        # z, _ = self.lstm(z.transpose(1,2))
        # print(z.shape)
        z = self.tr(z.transpose(1,2))
        # print('Trans = ', z.shape)
        # z = self.a(z.transpose(1,2))
        z = z.mean(1)
        return z

# class Accumulator(nn.Module):
#     def __init__(self, dim=128):
#         super().__init__()
#         self.dim = dim
#         self.transformer = nn.Mul
#         self.a = Attention(int(self.dim/1))

#     def forward(self, z):
#         # print(z.shape)
#         z, _ = self.lstm(z.transpose(1,2))
#         z = self.a(z.transpose(1,2))
#         return z


class Encoder2(nn.Module):
    def __init__(self, inchannel=120, channels=128):
        super().__init__()
        self.drop_prob = 0.1
        self.channels = channels

        self.k3 = nn.Sequential(
            nn.Conv1d(inchannel, self.channels, 3),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
            nn.Conv1d(self.channels, self.channels, 7),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
        )

        self.k5 = nn.Sequential(
            nn.Conv1d(inchannel, self.channels, 5),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
            nn.Conv1d(self.channels, self.channels, 5),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
        )

        self.k7 = nn.Sequential(
            nn.Conv1d(inchannel, self.channels, 7),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
            nn.Conv1d(self.channels, self.channels, 3),
            nn.BatchNorm1d(self.channels),
            nn.ReLU(),
        )


    def forward(self, x):
        z3 = self.k3(x)
        z5 = self.k5(x)
        z7 = self.k7(x)
        z = torch.cat([z3, z5, z7], 1)
        # print(z3.shape, z5.shape, z7.shape)

        return z


if __name__ == "__main__":
    x = torch.randn(2, 120, 100)
    model = Encoder2()
    z = model(x)
    print(z.shape)
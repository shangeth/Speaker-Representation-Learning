import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import numpy as np
import math
from transformers import Wav2Vec2Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class Encoder(nn.Module):
#     def __init__(self, channels=128):
#         super().__init__()
#         self.drop_prob = 0.1
#         self.channels = channels
#         self.encoder = nn.Sequential(
#             # SincConv_fast(self.channels, 251), 
#             nn.Conv1d(1, self.channels, 10, 5),
#             nn.BatchNorm1d(self.channels),
#             nn.ReLU(),

#             nn.Conv1d(self.channels, self.channels, 8, 4),
#             nn.BatchNorm1d(self.channels),
#             nn.ReLU(),

#             nn.Conv1d(self.channels, self.channels, 4, 2),
#             nn.BatchNorm1d(self.channels),
#             nn.ReLU(),

#             nn.Conv1d(self.channels, self.channels, 4, 2),
#             nn.BatchNorm1d(self.channels),
#             nn.ReLU(),

#             nn.Conv1d(self.channels, self.channels, 4, 2),
#             nn.BatchNorm1d(self.channels),
#             nn.ReLU(),

#             nn.Conv1d(self.channels, self.channels, 1, 1),
#             nn.BatchNorm1d(self.channels),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         return z


class Encoder(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_extractor
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        z = self.feature_extractor(x) # [Batch, D, N]
        z = self.transformer_encoder(z.transpose(1,2)) # [batch, N, D]
        return z

class Accumulator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.attention = Attention(self.dim)

    def forward(self, z):
        z = self.attention(z.transpose(1,2))
        return z
        

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
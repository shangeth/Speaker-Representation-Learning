import torch
import torch.nn as nn
import numpy as np
import wavencoder
from transformers import Wav2Vec2Model

class Encoder(nn.Module):
    def __init__(self, dim=128, nhead=8, nlayer=1):
        super().__init__()
        # self.feature_extractor = wavencoder.models.Wav2Vec(pretrained=True)
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        # for param in self.feature_extractor.feature_extractor.conv_layers[5:].parameters():
        #     param.requires_grad = True

        # self.batchNorm = nn.BatchNorm1d(512)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        x = self.feature_extractor(x.squeeze(1)) # [Batch, D, N]
        # x = self.batchNorm(x)
        # x = self.transformer_encoder(x.transpose(1,2)) # [batch, N, D]
        x = x.transpose(1,2)
        return x

class Accumulator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.attention = wavencoder.layers.SoftAttention(self.dim, self.dim)

    def forward(self, x):
        x = self.attention(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.classifier = nn.Sequential(
            nn.Linear(int(self.dim),  int(self.dim/4)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(int(self.dim/4), 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        y_hat = self.classifier(torch.cat([z1, z2], 1))
        return y_hat

class CenterLoss(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.distance_metric = nn.MSELoss()

    def forward(self, x):
        center = torch.mean(x, 1).unsqueeze(1)
        center = center.repeat(1, x.size(1), 1)
        return self.distance_metric(center, x)
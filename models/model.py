import torch
import torch.nn as nn
import numpy as np
from wavencoder.layers import SoftAttention


class Encoder(nn.Module):
    def __init__(self, dim=128, nhead=4, nlayer=1):
        super().__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").feature_extractor
        encoder_layer = nn.TransformerEncoderLayer(dim=dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

    def forward(self, x):
        x = self.feature_extractor(x.squeeze(1)) # [Batch, D, N]
        x = self.transformer_encoder(x.transpose(1,2)) # [batch, N, D]
        return x

class Accumulator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.attention = SoftAttention(self.dim)

    def forward(self, x):
        x = self.attention(x.transpose(1,2))
        return x


class CenterLoss(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.distance_metric = nn.MSELoss()

    def forward(self, x):
        center = torch.mean(x, 1).unsqueeze(1)
        center = center.repeat(1, x.size(1), 1)
        return self.distance_metric(center, x)

class Discriminator(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.classifier = nn.Sequential(
            nn.Linear(int(self.dim),  int(self.dim/2)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(int(self.dim/2), 1),
            nn.Sigmoid()
        )

    def forward(self, z1, z2):
        y_hat = self.classifier(torch.cat([z1, z2], 1))
        return y_hat
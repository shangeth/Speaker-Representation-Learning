from models.encoder import SincConv_fast, count_parameters
import torch 
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.drop_prob = 0.1
        self.channels = 128

        self.sincnet = nn.Sequential(
            SincConv_fast(self.channels, 251),
            nn.GroupNorm(32, self.channels),
            nn.ReLU(),)

        self.cnn = nn.Sequential(
            nn.Conv1d(1, self.channels, 251),
            nn.GroupNorm(32, self.channels),
            nn.ReLU(),)

        self.encoder = nn.Sequential(
            # nn.Conv1d(1, self.channels, 10, 5),
            # nn.Dropout(self.drop_prob),
            # nn.GroupNorm(32, self.channels),
            # nn.ReLU(),

            nn.Conv1d(2*self.channels, self.channels, 8, 4),
            nn.Dropout(self.drop_prob),
            nn.GroupNorm(32, self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 4, 2),
            nn.Dropout(self.drop_prob),
            nn.GroupNorm(32, self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 4, 2),
            nn.Dropout(self.drop_prob),
            nn.GroupNorm(32, self.channels),
            nn.ReLU(),

            nn.Conv1d(self.channels, self.channels, 4, 2),
            nn.Dropout(self.drop_prob),
            nn.GroupNorm(32, self.channels),
            nn.ReLU(),
        )

    def forward(self, x):
        sinc = self.sincnet(x)
        cnn = self.cnn(x)
        x = torch.cat([sinc, cnn], 1)
        z = self.encoder(x)
        return z


x = torch.randn(1, 1, 16000)
# enc = SincConv_fast(128, 251)
enc = Encoder()

z = enc(x)
# z2 = enc2(x)
# enc2 = torch.nn.Conv1d(1, 128, 251)
print(z.shape)
print(count_parameters(enc))

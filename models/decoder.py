import torch
import torch.nn as nn
from models.transformer import TransformerEncoder


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        self.upsample = nn.Sequential(

            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                stride=5,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(
                in_channels=256,
                out_channels=128,
                kernel_size=10,
                stride=2,
                padding=4,
                bias=False
            ),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(
                in_channels=128,
                out_channels=64,
                kernel_size=10,
                stride=2,
                padding=4,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=5,
                padding=0,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(params.dropout),

            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=2,
                kernel_size=64,
                stride=30,
                padding=17,
                bias=False
            )

        )

       
    def forward(self, x):
        bz = x.shape[0]
        #print(x.shape)
        x = x.view(bz*20, x.shape[-1], 1)
        x = self.upsample(x)
        return x.view(bz, 20, 2, 3000)

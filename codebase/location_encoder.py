import torch 
import torch.nn as nn 
import math 
import imageio.v2 as imageio
import rasterio as rio 
import numpy as np


class wrap_encoder(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x):
        coords_in_radians = torch.deg2rad(x)
        lon_cos = torch.cos(coords_in_radians[:, 0, ...]).unsqueeze(1)
        lon_sin = torch.sin(coords_in_radians[:, 1, ...]).unsqueeze(1)
        lat_cos = torch.cos(coords_in_radians[:, 0, ...]).unsqueeze(1)
        lat_sin = torch.sin(coords_in_radians[:, 1, ...]).unsqueeze(1)

        return torch.cat([lon_cos, lon_sin, lat_cos, lat_sin], dim=1).float()

class Siren2d(nn.Module):
    def __init__(
        self,
        dropout_rate=0.1,
        n=1,
        in_channels=4,
        hidden_channels=16,
        output_channels=1
    ):
        super(Siren2d, self).__init__()

        self.n_passes = n
        self.initial_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.dropout_rate = dropout_rate

        self.initial_conv = nn.Conv2d(self.initial_channels, self.hidden_channels, 3, padding='same')
        self.hidden_convs = nn.ModuleList(
            [nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding='same') for _ in range(self.n_passes - 1)]
            )
        self.final_conv = nn.Conv2d(self.hidden_channels, self.output_channels, 3, padding='same')
        self.dropout = nn.Dropout(self.dropout_rate)

        self.float()
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.dropout(x)
        x = torch.sin(x)
        
        for conv in self.hidden_convs:
            x = conv(x)
            x = self.dropout(x)
            x = torch.sin(x)
        
        x = self.final_conv(x)
        return x


class LocationEncoder(nn.Module):
    def __init__(self, dropout_rate, out_channels=1, n_sirens=1):
        super(LocationEncoder, self).__init__()
        self.encoder = wrap_encoder()
        self.siren = Siren2d(n=n_sirens, dropout_rate=dropout_rate, output_channels=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.siren(x)
        return x
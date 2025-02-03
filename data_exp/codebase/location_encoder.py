import torch 
import torch.nn as nn 
import math 


class wrap_encoder(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x):
        coords_in_radians = torch.deg2rad(x)
        lon_cos = torch.cos[coords_in_radians[0, ...]].unsqueeze(1)
        lon_sin = torch.sin[coords_in_radians[0, ...]].unsqueeze(1)
        lat_cos = torch.cos[coords_in_radians[1, ...]].unsqueeze(1)
        lat_sin = torch.sin[coords_in_radians[1, ...]].unsqueeze(1)

        return torch.cat([lon_cos, lon_sin, lat_cos, lat_sin], dim=1)

class 


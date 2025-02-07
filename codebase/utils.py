import torch  
import numpy as np 
import rasterio as rio


def retreive_xy_center(img_path):
    dataset = rio.open(img_path)
    bounds = dataset.bounds
    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.top + bounds.bottom) / 2
    center_coordinate = (center_x, center_y)
    return center_coordinate

def ce_per_class(ce_tensor, targets_tensor, n_classes): 
    class_losses = [] 
    for i in range(n_classes):
        mask = targets_tensor == i 
        if mask.sum() == 0: 
            class_losses.append(0.0)
        else:
            class_losses.append(torch.mean(ce_tensor[mask]).item())
    return class_losses
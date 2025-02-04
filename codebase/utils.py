import torch  
import numpy as np 

def ce_per_class(ce_tensor, targets_tensor, n_classes): 
    class_losses = [] 
    for i in range(n_classes):
        mask = targets_tensor == i 
        if mask.sum() == 0: 
            class_losses.append(0.0)
        else:
            class_losses.append(torch.mean(ce_tensor[mask]).item())
    return class_losses
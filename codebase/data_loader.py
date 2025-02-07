import os 
import numpy as np
import imageio.v2 as imageio
import rasterio as rio
from torch.utils.data import Dataset


def img_loader(img_path):
    img = np.array(imageio.imread(img_path), np.float32)
    return img


def retreive_xy(img_path):
    
    dataset = rio.open(img_path)
    idxs = np.indices((dataset.height, dataset.width))
    coords = np.vectorize(lambda x, y: dataset.transform * (x, y))(idxs[0], idxs[1])

    return np.array(coords)


class DFC25_TRAIN(Dataset):
    def __init__(
        self, 
        dataset_path, 
        dataset_file,
        loader,
        max_iters=10,
        transforms=None,
        crop_size=256,
        suffix='.tif'
    ):
    
        self.dataset_path = dataset_path
        with open(dataset_file, 'r') as f:
            self.dataset_list = [name.strip() for name in f]

        self.loader = loader
        
        self.crop_size = crop_size
        self.transforms = transforms
        self.suffix = suffix
        self.max_iters = max_iters

        if max_iters is not None:
            # get multiples of the dataset 
            self.dataset_list = self.dataset_list * int(np.ceil(float(self.max_iters) / len(self.dataset_list)))
            # select nsamples based on max_iters 
            self.dataset_list = self.dataset_list[0:max_iters]

        self.crop_size = crop_size
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        pre = self.loader(os.path.join(self.dataset_path, 'pre-event',  self.dataset_list[idx] + '_pre_disaster' + self.suffix))
        post = self.loader(os.path.join(self.dataset_path, 'post-event', self.dataset_list[idx] + '_post_disaster' + self.suffix)) 
        post = np.expand_dims(post, axis=2)
        img = np.concatenate([pre, post], axis=2) 
        
        mask = self.loader(os.path.join(self.dataset_path, 'target', self.dataset_list[idx] + '_building_damage' + self.suffix))

        coords = np.load(os.path.join(self.dataset_path, 'post-event-coords', self.dataset_list[idx] + '_post_disaster.npy'))
        coords = coords.transpose(1, 2, 0)

        if self.transforms: 
            transformed = self.transforms(image=img, mask=mask, coords=coords)
            img = transformed['image']
            mask = transformed['mask']
            coords = transformed['coords']

        return img.transpose(2, 0, 1), mask, coords.transpose(2, 0, 1)


class DFC25_VAL(Dataset):
    def __init__(
        self, 
        dataset_path, 
        dataset_file,
        loader,
        max_iters=10,
        transforms=None,
        suffix='.tif'
    ):
        self.max_iters = max_iters  
        self.dataset_path = dataset_path
        with open(dataset_file, 'r') as f:
            self.dataset_list = [name.strip() for name in f]

        self.dataset_list = np.random.choice(self.dataset_list, size=int(len(self.dataset_list) * 0.1), replace=False)


        if max_iters is not None:
            # get multiples of the dataset 
            self.dataset_list = self.dataset_list * int(np.ceil(float(self.max_iters) / len(self.dataset_list)))
            # select nsamples based on max_iters 
            self.dataset_list = self.dataset_list[0:max_iters]
        
        self.loader = loader
        self.transforms = transforms
        self.suffix = suffix
  
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        pre = self.loader(os.path.join(self.dataset_path, 'pre-event',  self.dataset_list[idx] + '_pre_disaster' + self.suffix))
        post = self.loader(os.path.join(self.dataset_path, 'post-event', self.dataset_list[idx] + '_post_disaster' + self.suffix)) 
        post = np.expand_dims(post, axis=2)
        img = np.concatenate([pre, post], axis=2) 
        
        mask = self.loader(os.path.join(self.dataset_path, 'target', self.dataset_list[idx] + '_building_damage' + self.suffix))
        
        coords = np.load(os.path.join(self.dataset_path, 'post-event-coords', self.dataset_list[idx] + '_post_disaster.npy'))
        coords = coords.transpose(1, 2, 0)

        if self.transforms: 
            transformed = self.transforms(image=img, mask=mask, coords=coords)
            img = transformed['image']
            mask = transformed['mask']
            coords = transformed['coords']

        return img.transpose(2, 0, 1), mask, coords.transpose(2, 0, 1)
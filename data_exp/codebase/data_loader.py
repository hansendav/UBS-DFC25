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


class DFC25_CU_LOADER(Dataset):
    def __init__(
        self, 
        dataset_path, 
        dataset_file,
        split='train',
        loader=img_loader,
        transforms=None,
        crop_size=256,
        suffix='.tif'
    ):
    
        self.dataset_path = dataset_path
        with open(dataset_file, 'r') as f:
            self.dataset_list = [name.strip() for name in f]
        
        self.split = split
        self.loader = loader
        self.crop_size = crop_size
        self.transforms = transforms
        self.suffix = suffix
    
    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, idx):
        pre = self.loader(os.path.join(self.dataset_path, 'pre-event',  self.dataset_list[idx] + '_pre_disaster' + self.suffix))
        post = self.loader(os.path.join(self.dataset_path, 'post-event', self.dataset_list[idx] + '_post_disaster' + self.suffix)) 
        mask = self.loader(os.path.join(self.dataset_path, 'target', self.dataset_list[idx] + '_building_damage' + self.suffix))
        post = np.expand_dims(post, axis=2)

        img = np.concatenate([pre, post], axis=2) 

        coords = retreive_xy(os.path.join(self.dataset_path, 'post-event', self.dataset_list[idx] + '_post_disaster' + self.suffix))
        coords = coords.transpose(1, 2, 0)

        if self.transforms: 
            transformed = self.transforms(image=img, mask=mask, coords=coords)
            img = transformed['image']
            mask = transformed['mask']
            coords = transformed['coords']

        return img, mask, coords
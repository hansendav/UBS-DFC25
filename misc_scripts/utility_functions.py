import rasterio as rio 
import numpy as np 
import time
import imageio.v2 as imageio


def img_loader(img_path):
    img = np.array(imageio.imread(img_path), np.float32)
    return img


def retreive_xy(img_path):
    
    dataset = rio.open(img_path)
    idxs = np.indices((dataset.height, dataset.width))
    coords = np.vectorize(lambda x, y: dataset.transform * (x, y))(idxs[0], idxs[1])

    return np.array(coords)






if __name__ == '__main__':
    img_path = '../../../dfc25_track2_trainval/train/post-event/bata-explosion_00000000_post_disaster.tif'
    
    
    start = time.time()
    img = img_loader(img_path)
    img_coords = retreive_xy(img_path)
    print(time.time() - start)

    print(img.shape)   
    print(img_coords.shape)
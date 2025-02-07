import sys 
sys.path.append('codebase')
import os 
import time 
import numpy as np
from tqdm import tqdm 

from data_loader import retreive_xy 


def main(path_to_files, output_path): 
    files = os.listdir(path_to_files)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with tqdm(total=len(files)) as pbar:
        for file in files: 
            save_name = file.split('.')[0] + '.npy'
            coords = retreive_xy(os.path.join(path_to_files, file))
            np.save(os.path.join(output_path, save_name), coords)
            pbar.update(1)

if __name__ == '__main__':
    path_to_files = '/home/gentleprotector/ubs_ws24/dfc25_track2_trainval/val/post-event/'
    output_path = '/home/gentleprotector/ubs_ws24/dfc25_track2_trainval/val/post-event-coords/'

    start = time.time()
    main(path_to_files, output_path)
    print(f'Finished in {time.time() - start} seconds')


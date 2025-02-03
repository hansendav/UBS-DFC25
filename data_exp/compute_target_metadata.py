import os 
import pandas as pd 
import numpy as np
import rasterio
import argparse





def main(path_to_dataset, output_file): 
    
    path_dataset = path_to_dataset

    train_post_files = os.listdir(path_dataset + 'train/post-event/')
    train_label_maps = os.listdir(path_dataset + 'train/target/')
    train_pre_files = os.listdir(path_dataset + 'train/pre-event/')

    target_dist_train = [] 

    for file in train_label_maps:
        with rasterio.open(os.path.join(path_dataset, 'train/target/' + file)) as src:
            img = src.read()
            min_lon, min_lat, max_lon, max_lat = src.bounds 

            if min_lon > max_lon: # take intern. date line into account (180/0) 
                max_lon += 360 
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            
            if center_lon > 180: # readjust to [-180, 180]
                center_lon -= 360
            
            targets_present = np.unique(img)
            background = np.count_nonzero(img == 0)
            building = np.count_nonzero(img)
            intact = np.count_nonzero(img == 1)
            damaged = np.count_nonzero(img == 2)
            destroyed = np.count_nonzero(img == 3)
            tile_post = file.replace('_building_damage', '_post_disaster')
            tile_pre = file.replace('_building_damage', '_pre_disaster')
            target_dist_train.append({
                'tile_mask': file, 
                'tile_pre': tile_pre,
                'tile_post': tile_post,
                'center_lon': center_lon,
                'center_lat': center_lat,
                'targets': np.array2string(targets_present),
                'background': background,
                'building': building,
                'intact': intact,
                'damaged': damaged,
                'destroyed': destroyed
            })

    target_dist_train = pd.DataFrame(target_dist_train)
    print(
        f'Initial dataframe created.\n'
    )

    # pixel per image ratio
    target_dist_train['background_ratio'] = target_dist_train['background'] / (1024**2)
    target_dist_train['building_ratio'] = target_dist_train['building'] / (1024**2)
    target_dist_train['intact_ratio'] = target_dist_train['intact'] / (1024**2)
    target_dist_train['damaged_ratio'] = target_dist_train['damaged'] / (1024**2)
    target_dist_train['destroyed_ratio'] = target_dist_train['destroyed'] / (1024**2)

    # pixel per building ratio 
    target_dist_train['intact_building_ratio'] = target_dist_train['intact'] / target_dist_train['building']
    target_dist_train['damaged_building_ratio'] = target_dist_train['damaged'] / target_dist_train['building']
    target_dist_train['destroyed_building_ratio'] = target_dist_train['destroyed'] / target_dist_train['building']

    target_dist_train['place'] = target_dist_train.apply(lambda x: x['tile_post'].split('-')[0], axis=1)
    target_dist_train['disaster'] = target_dist_train.apply(lambda x: x['tile_post'].split('-')[1].split('_')[0], axis=1)


    target_dist_train['tile_name'] = target_dist_train['tile_mask'].apply(lambda x: x.split('_building')[0]) 
    
    target_dist_train.to_csv(output_file, index=False)

    print(
        f'Final dataframe saved to {output_file}\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str, default='../../dfc25_track2_trainval/')
    parser.add_argument('--output_file', type=str, default='./target_metadata.csv')
    args = parser.parse_args()
    main(args.path_to_dataset, args.output_file)
#!/bin/bash

# Define the arguments
NORM_MEAN="88.26 83.83 75.40 57.39"
NORM_STD="38.06 31.84 28.87 24.43"
CROP_SIZE=256
TRAIN_DATA_PATH="/home/gentleprotector/ubs_ws24/dfc25_track2_trainval/train/"
TRAIN_FILE="/home/gentleprotector/ubs_ws24/BRIGHT/dfc25_benchmark/dataset/splitname/train_setlevel.txt"
VAL_DATA_PATH="/home/gentleprotector/ubs_ws24/dfc25_track2_trainval/train/"
VAL_FILE="/home/gentleprotector/ubs_ws24/BRIGHT/dfc25_benchmark/dataset/splitname/holdout_setlevel.txt"
MAX_ITERS=27863
BATCH_SIZE=4
LR=0.001
CLASS_WEIGHTS="0.001 0.02 0.58972941 0.376140632"
LABEL_SMOOTHING=0.0
SAVE_DIR="../../experiments/"
RUN_NAME="unet_no_location"
RESUME="/path/to/checkpoint"

# Run the script
python ../train_dfc.py \
    --norm_mean $NORM_MEAN \
    --norm_std $NORM_STD \
    --crop_size $CROP_SIZE \
    --train_data_path $TRAIN_DATA_PATH \
    --max_iters $MAX_ITERS \
    --train_file $TRAIN_FILE \
    --val_data_path $VAL_DATA_PATH \
    --val_file $VAL_FILE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --class_weights $CLASS_WEIGHTS \
    --label_smoothing $LABEL_SMOOTHING \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME \
    #--resume $RESUME


#!/bin/bash

# Define the arguments
CROP_SIZE=256
TRAIN_DATA_PATH="/home/gentleprotector/ubs_ws24/dfc25_track2_trainval/train/"
TRAIN_FILE="/home/gentleprotector/ubs_ws24/BRIGHT/dfc25_benchmark/dataset/splitname/train_setlevel.txt"
VAL_DATA_PATH="/home/gentleprotector/ubs_ws24/dfc25_track2_trainval/train/"
VAL_FILE="/home/gentleprotector/ubs_ws24/BRIGHT/dfc25_benchmark/dataset/splitname/holdout_setlevel.txt"
NUM_EPOCHS=100
BATCH_SIZE=4
NUM_WORKERS=4
LR=0.0001
CLASS_WEIGHTS="[1.18, 7.46, 149.29, 95.22]"
SAVE_DIR="../../experiments/"
RUN_NAME="baseline_unetloc_center_loc"

# Run the script
python ../train_baseline_unetloc.py \
    --crop_size $CROP_SIZE \
    --dataset_train_path $TRAIN_DATA_PATH \
    --train_file $TRAIN_FILE \
    --dataset_val_path $VAL_DATA_PATH \
    --num_workers $NUM_WORKERS \
    --val_file $VAL_FILE \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --save_dir $SAVE_DIR \
    --run_name $RUN_NAME


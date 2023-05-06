#!/bin/sh

python split_dataset.py
python create_tf_record.py -x ../train_val_dataset/train/labels -l ./label_map.pbtxt -o ../train_val_dataset/train.record -i ../train_val_dataset/train/images
python create_tf_record.py -x ../train_val_dataset/test/labels -l ./label_map.pbtxt -o ../train_val_dataset/test.record -i ../train_val_dataset/test/images

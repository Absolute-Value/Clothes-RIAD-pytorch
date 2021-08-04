#!/bin/sh
echo "Start"
for seed in 999 888 777
do
    python3 train.py --dataset other --data_type wrench --batch_size 16 --epochs 300 --img_size 256 --init_type normal --lr 0.0001 --seed $seed
done
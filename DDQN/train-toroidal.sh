#!/bin/bash

EPSILON=2
NUM_TRAIN=50000

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 1001 --lr 0.3\
                --q_net_iter 128 --batch_size 128 --mem_cap 2048\
                --num_conv 12 --gamma 0.7 --net_type 'toroidal'

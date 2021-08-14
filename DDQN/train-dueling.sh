#!/bin/bash

EPSILON=1
NUM_TRAIN=1000

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 500 --lr 0.7\
                --q_net_iter 128 --batch_size 1024 --mem_cap 2048\
                --num_conv 12 --gamma 0.7 --net_type 'dueling'

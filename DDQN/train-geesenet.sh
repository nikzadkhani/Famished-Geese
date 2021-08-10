#!/bin/bash

EPSILON=1
NUM_TRAIN=5000

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 1000 --lr 0.7\
                --q_net_iter 4 --batch_size 1024 --mem_cap 2048\
                --num_conv 12 --gamma 0.7 --net_type 'geese'

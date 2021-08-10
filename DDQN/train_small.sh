#!/bin/bash

EPSILON=1
NUM_TRAIN=100

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 50 --lr 0.3\
                --q_net_iter 50 --batch_size 25 --mem_cap 50\
                --num_conv 6

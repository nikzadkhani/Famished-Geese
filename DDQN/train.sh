#!/bin/bash

EPSILON=2
NUM_TRAIN=10000

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 200 --lr 0.3\
                --q_net_iter 128 --batch_size 128 --mem_cap 2048\
                --num_conv 12 --gamma 0.7 --net_type 'dueling'

#!/bin/bash

EPSILON=0.7
NUM_TRAIN=60

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 10000 --lr 0.0000003\
                --q_net_iter 10 --batch_size 10 --mem_cap 10\
                --num_conv 14

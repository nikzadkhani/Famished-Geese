#!/bin/bash

EPSILON=0.3
NUM_TRAIN=100000

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 10000 \
                --q_net_iter 512

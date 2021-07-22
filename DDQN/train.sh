#!/bin/bash

EPSILON=0.01
NUM_TRAIN=2000

python3 main.py --eps $EPSILON --num_episodes $NUM_TRAIN \
                --save_path './saved_models/' --testing 0 \
                --save_interval 1024 \
                --q_net_iter 216

#!/bin/bash

EPSILON=0
NUM_TEST=200

python3 main.py --eps $EPSILON --num_episodes $NUM_TEST \
                --save_path 'saved_models/' --testing 1 \
                --save_interval 1024 \
                --q_net_iter 216 --test_model 'ddqn-final-10000.pt'

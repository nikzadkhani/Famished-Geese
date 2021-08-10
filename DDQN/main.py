from model import DDQN

from train import train
from test import test
from options import get_parser
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

NUM_STATES = 77
NUM_ACTIONS = 4
SEED = 42


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    opt = get_parser().parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    if opt.testing:
        print('Starting testing...\n')
        ddqn = DDQN(NUM_STATES, NUM_ACTIONS, opt.num_conv, opt.eps, opt)
        ddqn.load(opt.save_path + opt.test_model)
        print(opt.render)
        test(ddqn, opt.num_episodes, render=True)
    else:
        print('Starting training...\n')
        ddqn = DDQN(NUM_STATES, NUM_ACTIONS, opt.num_conv, opt.eps, opt)
        train(ddqn, opt.num_episodes, opt)
        # print(ddqn.losses)


if __name__ == '__main__':
    main()

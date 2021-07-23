from model import DDQN

from train import train
from test import test
from options import get_parser
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
    test(opt.num_episodes)
  else:
    print('Starting training...\n')
    ddqn = DDQN(NUM_STATES, NUM_ACTIONS, opt.eps, opt)
    train(ddqn, opt.num_episodes)


if __name__ == '__main__':
  main()
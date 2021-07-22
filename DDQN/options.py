from argparse import ArgumentParser

# DEFAULT PARAMETERS
BATCH_SIZE = 128
LR = 1e-5
GAMMA = 0.9
EPISILON = 0.9
MEMORY_CAPACITY = 2000
NUM_EPISODES = 1000
NUM_TEST = 100
SAVE_INTERVAL = 100
Q_NETWORK_ITERATION = 100

def get_parser() -> ArgumentParser:
  parser = ArgumentParser()
  parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size; default=%i' % BATCH_SIZE)
  parser.add_argument('--lr', default=LR, type=float, help='learning rate; default=%i' % LR)
  parser.add_argument('--gamma', default=GAMMA, type=float, help='discount factor; default=%i' % GAMMA)
  parser.add_argument('--eps', default=EPISILON, type=float, help='epsilon/exploration coeff; default=%i' % EPISILON)
  parser.add_argument('--mem_cap', default=MEMORY_CAPACITY, type=int, help='memory capacity; default=%i' % MEMORY_CAPACITY)
  parser.add_argument('--num_episodes', default=NUM_EPISODES, type=int, help='number of episodes; default=%i' % NUM_EPISODES)
  parser.add_argument('--num_test', default=NUM_TEST, type=int, help='number of test episodes; default=%i' % NUM_TEST)
  parser.add_argument('--save_path', required=True, help='where all models will be saved')
  parser.add_argument('--q_net_iter', default=Q_NETWORK_ITERATION, type=int, help='number of iterations to wait before we update the target network')
  parser.add_argument('--save_interval', type=int,default=SAVE_INTERVAL, help='how often to save the model to the save_path; default=%i' % SAVE_INTERVAL)
  parser.add_argument('--testing', required=True, help='set 1 if testing old model and 0 if training new model')
  parser.add_argument('--test_model', help='file name of test model in save_path')
  return parser
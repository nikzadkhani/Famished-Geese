from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, \
Action, row_col
from vector import create_state_vector, get_action
from model import DDQN
from argparse import Namespace

NUM_STATES = 77
NUM_ACTIONS = 4
EPSILON = 0
ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']

OPTIONS = Namespace(batch_size=128, lr=1e-05, gamma=0.9, eps=0.01, mem_cap=2000,\
                    num_episodes=2000, num_test=100, save_path='./saved_models/',\
                    q_net_iter=100, save_interval=100, testing='1', test_model='ddqn-final-10000.pt')

ddqn = DDQN(NUM_STATES, NUM_ACTIONS, EPSILON, OPTIONS)
ddqn.load(ddqn.opt.save_path + ddqn.opt.test_model)


def agent(obs_dict, config_dict):
  observation = Observation(obs_dict)
  configuration = Configuration(config_dict)
  
  state_vector = create_state_vector(observation)
  action_index = ddqn.choose_action(state_vector)
  action = get_action(action_index)

  return action




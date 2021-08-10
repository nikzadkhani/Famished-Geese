from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, \
    Action, row_col
from vector import create_state_vector, get_action
from model import DDQN
from options import read_namespace
import os

NUM_STATES = 77
NUM_ACTIONS = 4
EPSILON = 0
ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']

OPTIONS_JSON = 'test_options.json'
opt = read_namespace(OPTIONS_JSON)
ddqn = DDQN(NUM_STATES, NUM_ACTIONS, EPSILON, opt)
model_fpath = opt.save_path + opt.test_model

if os.path.isfile(model_fpath):
    ddqn.load(model_fpath)
    # for i in ddqn.eval_net.parameters():
    #   print(i)

    # print(sum(p.numel() for p in ddqn.eval_net.parameters()))
# else:
#   raise FileNotFoundError("DDQN torch model %s does not exist" % model_fpath)


def agent(obs_dict, config_dict):
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)

    state_vector = create_state_vector(observation)
    action_index = ddqn.choose_action(state_vector)
    action = get_action(action_index)

    return action

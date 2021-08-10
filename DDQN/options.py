from argparse import ArgumentParser, Namespace
from json import load

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
RENDER = 0
NUM_CONV = 2
DEFAULT_NET = 'dueling'


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=BATCH_SIZE,
        type=int,
        help='batch size; default=%i' %
        BATCH_SIZE)
    parser.add_argument(
        '--lr',
        default=LR,
        type=float,
        help='learning rate; default=%i' %
        LR)
    parser.add_argument(
        '--gamma',
        default=GAMMA,
        type=float,
        help='discount factor; default=%i' %
        GAMMA)
    parser.add_argument(
        '--eps',
        default=EPISILON,
        type=float,
        help='epsilon/exploration coeff; default=%i' %
        EPISILON)
    parser.add_argument(
        '--mem_cap',
        default=MEMORY_CAPACITY,
        type=int,
        help='memory capacity; default=%i' %
        MEMORY_CAPACITY)
    parser.add_argument(
        '--num_episodes',
        default=NUM_EPISODES,
        type=int,
        help='number of episodes; default=%i' %
        NUM_EPISODES)
    parser.add_argument(
        '--num_test',
        default=NUM_TEST,
        type=int,
        help='number of test episodes; default=%i' %
        NUM_TEST)
    parser.add_argument(
        '--save_path',
        required=True,
        help='where all models will be saved')
    parser.add_argument(
        '--q_net_iter',
        default=Q_NETWORK_ITERATION,
        type=int,
        help='number of iterations to wait before we update the target network')
    parser.add_argument(
        '--save_interval',
        type=int,
        default=SAVE_INTERVAL,
        help='how often to save the model to the save_path; default=%i' %
        SAVE_INTERVAL)
    parser.add_argument(
        '--testing',
        type=int,
        required=True,
        help='set 1 if testing old model and 0 if training new model')
    parser.add_argument(
        '--test_model',
        help='file name of test model in save_path')
    parser.add_argument(
        '--render',
        type=bool,
        default=RENDER,
        help='if 0 will not render anything else will render each testing example to browser; default=%i' %
        RENDER)
    parser.add_argument(
        '--num_conv',
        type=int,
        default=NUM_CONV,
        help='The number of convolution layers that will encode the grid')
    parser.add_argument(
        '--net_type',
        required=True,
        type=str,
        default=DEFAULT_NET,
        help='which neural net to use to train/test;\
        ie. dueling, toroidal, vanilla')
    return parser


def read_namespace(fpath: str) -> Namespace:
    """
    Read in the argparse.Namespace from file rather than cmd line for submission to kaggle
    Most of these arguments are unused so this is mostly just to make the code work, eventually
    need to fix this garbage...
    Keeping the data namespace to be consisting with training...
    :param fpath: The filepath to the arguments settings in json format
    :returns: Argparse namespace object
    """
    with open(fpath, 'r') as f:
        d = load(f)
    namespace = Namespace()
    namespace_d = vars(namespace)  # returns dict attribute of namespace
    namespace_d.update(d)
    return namespace

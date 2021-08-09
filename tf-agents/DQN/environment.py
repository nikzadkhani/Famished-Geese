
from kaggle_environments.envs.hungry_geese.hungry_geese import (Observation, 
                                                                Configuration, 
                                                                Action, 
                                                                row_col, 
                                                                translate, 
                                                                greedy_agent)
from kaggle_environments import evaluate, make, utils

import numpy as np
import random
from tqdm import tqdm 
import abc
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
tf.compat.v1.enable_v2_behavior()

class TFHungryGoose(py_environment.PyEnvironment):
    def __init__(self):
        env = make("hungry_geese")
        self.trainer = env.train([None, greedy_agent, greedy_agent, greedy_agent])
        obs = self.trainer.reset()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1, 7, 11, 1), dtype=np.float32, minimum=0, maximum=10, name='observation')
        self._state = self.create_grid_from_geese_position(obs)
        self._episode_ended = False
        self.action_name_mapping = {0: 'NORTH', 1: 'SOUTH', 2: 'EAST', 3: 'WEST'}
    def create_grid_from_geese_position(self, obs, grid_cols=11, grid_rows=7):
        geese_position = obs.geese
        foods = obs.food
        my_index = obs.index
        matrix = np.zeros((grid_rows, grid_cols))
        goose_id = [1, 2, 3, 4]
        for i, goose_position in enumerate(geese_position):
            for j, pos in enumerate(goose_position):
                row, col = row_col(pos, grid_cols)
                if j == 0:
                    if i!=my_index:
                        matrix[row][col] = 6
                    else:
                        matrix[row][col] = 7
                else:
                    matrix[row][col] = goose_id[i]   
        np.put(matrix, foods, [5])
        return matrix.reshape(1, 7, 11, 1).astype('float32')
    def action_spec(self):
        return self._action_spec
    def observation_spec(self):
        return self._observation_spec
    def _reset(self):
        obs = self.trainer.reset()
        self._state = self.create_grid_from_geese_position(obs)
        self._episode_ended = False
        return ts.restart(self._state)
    def __reward_manager(self, reward, step, geese): 
        if step == 1 and (reward != 0): 
            return 50
        elif (reward == 0) or (len(geese[0])==0): 
            return -1000
        elif (max([len(goose) for goose in geese[1:]]) == 0) and (reward != 0): 
            return 1000
        elif (reward%100)==0: 
            return 50
        else: 
            return 100
        
    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        action = self.action_name_mapping[int(action)]
        obs, reward, self._episode_ended, info = self.trainer.step(action)
        reward = self.__reward_manager(reward, obs.step, obs.geese)
        self._state = self.create_grid_from_geese_position(obs)
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)

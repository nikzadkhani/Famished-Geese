# train.py
"""
!apt-get update
!apt-get install libsdl2-gfx-dev libsdl2-ttf-dev
!sudo apt-get install -y xvfb ffmpeg python-opengl
!pip3 install pyglet
# # Make sure that the Branch in git clone and in wget call matches !!
!git clone -b v2.9 https://github.com/google-research/football.git
!mkdir -p football/third_party/gfootball_engine/lib
!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.8.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
!cd football && GFOOTBALL_USE_PREBUILT_SO=1 python3 -m pip install .
!pip install 'imageio==2.4.0'
!pip install 'xvfbwrapper==0.2.9'
!pip3 install tf_agents
!pip3 install kaggle-environments
!pip install 'imageio==2.4.0'
!pip install 'xvfbwrapper==0.2.9'
"""

# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from datetime import datetime
from kaggle_environments.envs.hungry_geese.hungry_geese import (Observation,Configuration,Action,row_col,translate,greedy_agent)
from kaggle_environments import evaluate, make, utils
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, Flatten,LeakyReLU)
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import (py_environment,tf_environment,tf_py_environment,utils,wrappers,suite_gym)
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver,random_tf_policy
from tf_agents.trajectories import time_step as ts,trajectory
from tf_agents.specs import array_spec,tensor_spec
from tf_agents.networks import sequential
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train.utils import strategy_utils, train_utils
from tf_agents.utils import common
from tqdm import tqdm
from packaging import version
import shutil
tf.compat.v1.enable_v2_behavior()
import base64
import imageio
import io
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import tensorflow as tf
import zipfile
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
tf.compat.v1.enable_v2_behavior()
#tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())



#Environment Wrapper

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
                    if i != my_index:
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
      #The reward is calculated as the current turn + goose length.
      if step == 1 and (reward != 0): # first step and survived, return only survive reward
            return 50
      elif(reward == 0) or (len(geese[0])==0): # you loose, hence large neg reward
            return -1000
        # check if you won or not
      elif (max([len(goose) for goose in geese[1:]]) == 0) and (reward != 0): #you just won
            return 1000
      elif (reward%100)==0: # you survived but not won nor ate, hence only survive reward
            return 50
      else: # you survived and ate, hence
            return 10
      "return step+ len(geese[0])"
      """ old reward version
      if step == 1 and (reward != 0):
          return 50
      elif (reward == 0) or (len(geese[0]) == 0):
          return -400
      elif (max([len(goose) for goose in geese[1:]]) == 0) and (reward != 0):
          return 1000
      elif (reward % 100) == 0:
          return 50
      else:
          return 100"""
    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        print(action)
        action = self.action_name_mapping[int(action)]
        print(action)
        obs, reward, self._episode_ended, info = self.trainer.step(action)
        reward = self.__reward_manager(reward, obs.step, obs.geese)
        self._state = self.create_grid_from_geese_position(obs)
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=1.0)
print("Compiled")

train_py_env=TFHungryGoose()
eval_py_env=TFHungryGoose()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

model = sequential.Sequential([
    Conv2D(64, kernel_size=5, activation=LeakyReLU()),
    Conv2D(32, kernel_size=3, activation=LeakyReLU()),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(4, activation=None)])

q_net =model




 
 
 
 
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
global_step = tf.compat.v1.train.get_or_create_global_step()
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=global_step)
agent.initialize()
replay_buffer_max_length=1000000
collect_steps_per_iteration=50
batch_size=50
# more to add

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)
# Initial data collection
collect_driver.run()
# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
iterator = iter(dataset)
def compute_avg_return(environment, policy, num_episodes=100):
  total_return, won = 0.0, 0
  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    while not time_step.is_last():
      action_step = policy.action(time_step)
      print(action_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return
    if episode_return > 1000:
        won+=1
  avg_return = total_return / num_episodes
  return avg_return, won / num_episodes
best_return = 0
def save_model_if_best(agent, avg_return,global_step,train_checkpointer):
    global best_return
    if avg_return > best_return:
        train_checkpointer.save(global_step)
        print(f'saved model, best return={avg_return:,.0f}')
        best_return = avg_return
cwd=os. getcwd()
checkpoint_dir = os.path.join(cwd, 'checkpoint-v2-test')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step)
# Training Code
agent.train = common.function(agent.train)
def train_one_iteration(train_checkpointer=train_checkpointer,global_step=global_step):
    #print("Training")
  # Collect a few steps using collect_policy and save to the replay buffer.
    collect_driver.run()
    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)
    #print("Training")
    iteration = agent.train_step_counter.numpy()
    #print("Training")
    if(iteration % 10 ==0):
        #print ('iteration: {0} loss: {1}'.format(iteration, train_loss.loss))
        avg_return,win_rate=compute_avg_return(environment=eval_env,policy=agent.policy,num_episodes=100)
        #print("Average score: ",avg_return.numpy()[0],"\n Win Rate: ",win_rate)
        save_model_if_best(agent, avg_return.numpy()[0],global_step,train_checkpointer)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.loss, step=iteration)
            tf.summary.scalar('win rate', win_rate, step=iteration)
            tf.summary.scalar('score',avg_return.numpy()[0], step=iteration) 
    else:
        pass

def train(num_iterations=100):
  for i in range(num_iterations):
    train_one_iteration() 
training_logs=os.path.join(cwd,"logs-v2-big-job")
try: 
    os.mkdir(training_logs,exists_ok=True)
except:
    pass
print("start of training")
try:
    train_summary_writer = tf.summary.create_file_writer(training_logs)
except:
    pass
train(300000)
print("Finished Training")
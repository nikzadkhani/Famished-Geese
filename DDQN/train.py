import kaggle_environments
import torch
import os
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action
import numpy as np

from model import DDQN
from vector import create_state_vector, get_action
from tqdm import trange


env = make("hungry_geese")
# Train against greedy
trainer = env.train([None, "greedy", "greedy", "greedy"])
trainer.reset()

ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']


def train_single_episode(ddqn: DDQN):
  """
  Trains single episode
  :returns: total reward for the episode and whether or not our goose was the victor
  """
  total_episode_reward = 0
  step_counter = 0
  done = False

  state = trainer.reset()
  state_vector = create_state_vector(state)


  while not done or step_counter == 200:
    action_vector = ddqn.choose_action(state_vector)
    action = get_action(action_vector)

    # Take our action
    new_state, reward, done, _ = trainer.step(action)
    new_state_vector = create_state_vector(new_state)

    # Store for experience replay
    ddqn.store_transition(state_vector, action_vector, reward, new_state_vector)
    total_episode_reward += reward

    ddqn.learn()

    step_counter += 1
    state_vector = new_state_vector
  
  return total_episode_reward, len(new_state.geese[0]) != 0

def train(ddqn: DDQN, num_episodes):
  rewards = []
  wins = []
  for i in trange(num_episodes):
    ddqn.ep_decay(num_episodes, i)
    reward, did_win = train_single_episode(ddqn)
    rewards.append(reward)
    wins.append(did_win)

    # Save model, rewards, and wins every save interval
    if i % ddqn.opt.save_interval == 0:
      ddqn_path = os.path.join(ddqn.opt.save_path, 'ddqn-' + str(i).zfill(5))
      reward_path = os.path.join(ddqn.opt.save_path, 'rewards-' + str(i).zfill(5))
      wins_path = os.path.join(ddqn.opt.save_path, 'wins-' + str(i).zfill(5))
      print("saving model at episode %i in save_path=%s" % (i, ddqn_path))
      ddqn.save(ddqn_path + '.pt')
      np.savetxt(reward_path, rewards, fmt="%4.1d")
      np.savetxt(wins_path, wins, fmt="%1d")

  # save final info
  ddqn_path = os.path.join(ddqn.opt.save_path, 'ddqn-final-' + str(num_episodes).zfill(5))
  reward_path = os.path.join(ddqn.opt.save_path, 'rewards-final-' + str(num_episodes).zfill(5))
  wins_path = os.path.join(ddqn.opt.save_path, 'wins-final-' + str(num_episodes).zfill(5))
  print("saving final model")
  ddqn.save(ddqn_path + '.pt')
  np.savetxt(reward_path, rewards, fmt="%4.1d")
  np.savetxt(wins_path, wins, fmt="%1d")

  

  return rewards, wins


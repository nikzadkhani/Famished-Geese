import numpy as np
import torch

from model import DDQN
from kaggle_environments import make
from tqdm import trange
from agent import agent


env = make("hungry_geese", debug=True)

def is_win(observations) -> bool:
  last_observation = observations[-1]

  # if goose is dead
  if len(last_observation[0]) == 0:
    return False
  
  best_length = 0
  for goose in last_observation:
    if len(goose) > best_length:
      best_length = len(goose)
  
  # return if our goose was the best
  return best_length == len(last_observation[0])



def test_single_episode(render=False):
  step_counter = 0
  done = False

  observations = env.run([agent, "greedy", "greedy", "greedy"])
  return is_win(observations), len(observations[-1][0])
  
def test(num_episodes):
  num_wins = 0
  goose_lengths = []
  for i in trange(num_episodes):
    was_win, player_goose_length = test_single_episode()

    if was_win:
      num_wins +=1
    
    goose_lengths.append(player_goose_length)
  
  s = "Num Wins/Num Episodes: {}/{} {}\nAverage Goose Length: {}\nMin Goose Length: {}\nMax Goose Length: {}\n"
  print(s.format(num_wins, num_episodes, num_wins/num_episodes, np.mean(goose_lengths), min(goose_lengths), max(goose_lengths)))
    
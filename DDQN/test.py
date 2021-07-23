import sys
import numpy as np
from graphics import render_env
from model import DDQN
from kaggle_environments import make
from tqdm import trange
from agent import agent


env = make("hungry_geese")


def test_single_episode(render=False):
  observations = env.run([agent, "greedy", "greedy", "greedy"])

  if render:
    render_env(env)
    for i,o in enumerate(observations):
      print("Iteration :", i)
      for observed in o:
        print(observed)
      print()

    response = input("Press any key to continue or q to stop the program:\n")
    print("Player 0 is the white goose")
    if response =="q":
      quit()


  last_observation = observations[-1]

  final_player_goose = observations[-1][0].observation.geese[0]

  # if goose is dead
  if len(final_player_goose) == 0:
    return False, len(final_player_goose)
  
  best_length = 0
  for goose in last_observation:
    if len(goose) > best_length:
      best_length = len(goose)
  
  # return if our goose was the best
  return best_length == len(final_player_goose), len(final_player_goose)
  
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
    
import kaggle_environments
import numpy as np

from model import DDQN

ACTIONS = ['NORTH', 'SOUTH', 'EAST', 'WEST']

def create_state_vector(state: kaggle_environments.utils.Struct) -> np.array:
  """
  converts state struct to 1 by 77 vector
  the integers in the vector are as follows:

  0 -> player head
  1 -> player body
  2 -> enemy head
  3 -> enemy body  
  4 -> food
  """
  # make sure we are playing with more than one goose
  assert(len(state.geese) > 1)

  PLAYER_HEAD = 0
  PLAYER_BODY = -1
  ENEMY_HEAD = -1
  ENEMY_BODY = -1
  FOOD = 1

  vector = np.zeros((77))
  
  player_goose = state.geese[state.index]
  enemy_geese = state.geese[:state.index] + state.geese[state.index+1:]

  # add player to vector
  # if player is alive
  if len(player_goose) > 0:
    vector[player_goose[0]] = PLAYER_HEAD
    for body_position in player_goose[1:]:
      vector[body_position] = PLAYER_BODY

  # add enemies to vector
  for enemy_goose in enemy_geese:
    # is alive
    if len(enemy_goose) > 0:
      vector[enemy_goose[0]] = ENEMY_HEAD
      for body_position in enemy_goose[1:]:
        vector[body_position] = ENEMY_BODY

  # add food to vector
  for food_position in state.food:
    vector[food_position] = FOOD

  return vector
  
def create_action_vector(action: str) -> np.array:
  assert(action in ACTIONS)
  return (np.array(ACTIONS) == action).astype(int)

def get_action(action_index: int) -> str:
  assert(0 <= action_index < len(ACTIONS))
  return ACTIONS[action_index]
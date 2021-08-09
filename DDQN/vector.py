import kaggle_environments
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col
import numpy as np
import torch
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
  PLAYER_BODY = 1
  ENEMY_HEAD = 2
  ENEMY_BODY = 3
  FOOD = 4

  vector = np.zeros((5, 77))
  
  player_goose = state.geese[state.index]
  enemy_geese = state.geese[:state.index] + state.geese[state.index+1:]

  # add player to vector
  # if player is alive
  if len(player_goose) > 0:
    vector[PLAYER_HEAD][player_goose[0]] = 1
    for body_position in player_goose[1:]:
      vector[PLAYER_BODY][body_position] = 1

  # add enemies to vector
  for enemy_goose in enemy_geese:
    # is alive
    if len(enemy_goose) > 0:
      vector[ENEMY_HEAD][enemy_goose[0]] = 1
      for body_position in enemy_goose[1:]:
        vector[ENEMY_BODY][body_position] = 1

  # add food to vector
  for food_position in state.food:
    vector[FOOD][food_position] = 1

  return torch.Tensor(vector.reshape(-1, 5, 7, 11))
  
def create_action_vector(action: str) -> np.array:
  assert(action in ACTIONS)
  return (np.array(ACTIONS) == action).astype(int)

def get_action(action_index: int) -> str:
  assert(0 <= action_index < len(ACTIONS))
  return ACTIONS[action_index]


def create_norm_state_vector(state: kaggle_environments.utils.Struct, previous_state: kaggle_environments.utils.Struct) -> np.array:
  def centerize(b):
    print(b[0].shape)
    dy, dx = np.where(b[0])
    # print(dy.shape, dx.shape)
    centerize_y = (np.arange(0,7)-3+dy[0])%7
    centerize_x = (np.arange(0,11)-5+dx[0])%11
    
    b = b[:, centerize_y,:]
    b = b[:, :,centerize_x]
    
    return b


  vector = np.zeros((17, 7 * 11), dtype=np.float32)

  for player_number, geese_positions in enumerate(state.geese):
      # head position
      for position in geese_positions[:1]:
          vector[0 + (player_number - state.index) % 4, position] = 1
      # tip position
      for position in geese_positions[-1:]:
          vector[4 + (player_number - state.index) % 4, position] = 1
      # whole position
      for position in geese_positions:
          vector[8 + (player_number - state.index) % 4, position] = 1
          
  if previous_state is not None:
    for player_number, geese_positions in  enumerate(previous_state.geese):
      vector[12 + (player_number - state.index) % 4, position] = 1

  # food
  for position in state.food:
      vector[16, position] = 1
      
  vector = vector.reshape(-1, 7, 11)
  # print(vector.shape)
  # vector = centerize(vector)
  vector = np.transpose(vector, (1,2,0))

  return torch.Tensor(vector).reshape(-1, 17, 7, 11)

def get_valid_actions(last_action: str):
  """ last_action = [left, straight, right] relative to geese
  """
  # Facing North
  if last_action == 'NORTH':
    valid_actions = ('WEST', 'NORTH', 'EAST')
  elif last_action == 'WEST':
    valid_actions = ('SOUTH', 'WEST', 'NORTH')
  elif last_action == 'EAST':
    valid_actions = ('NORTH', 'EAST', 'SOUTH')
  else:
    valid_actions = ('EAST', 'SOUTH', 'WEST')

  return valid_actions

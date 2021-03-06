import os
import tracemalloc
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col

from model import DDQN
from vector import create_norm_state_vector, get_action
from tqdm import trange
import numpy as np
from utils import display_top

import tracemalloc

env = make("hungry_geese")
# Train against greedy
trainer = env.train([None, "greedy", "greedy", "greedy"])
trainer.reset()

ACTIONS = ['NORTH', 'SOUTH', 'WEST', 'EAST']
NUM_COLS = 11
NUM_ROWS = 7

# tracemalloc.start()


def calculate_reward(state, new_state) -> int:
    reward = 0

    goose_pos = state.geese[0]
    new_goose_pos = new_state.geese[0]

    goose_len = len(goose_pos)
    new_goose_len = len(new_goose_pos)

    # If goose dies, punish corpse
    if new_goose_len == 0:
        return -1000

    # If goose got bigger, give pat on back
    if new_goose_len > goose_len:
        return 500

    def manhattan_distance(x1, y1, x2, y2): return abs(x1 - x2) + abs(y1 - y2)

    def toroidal_manhattan_distance(x1, y1, x2, y2):
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        if dx > NUM_COLS // 2:
            dx = NUM_COLS - dx

        if dy > NUM_ROWS // 2:
            dy = NUM_ROWS - dy

        return dx + dy

    # Convert goose heads to cartesian coordinates
    goose_x, goose_y = row_col(goose_pos[0], NUM_COLS)
    new_goose_x, new_goose_y = row_col(goose_pos[0], NUM_COLS)

    closest_food = float("inf")
    for food_pos in state.food:
        food_x, food_y = row_col(food_pos, NUM_COLS)
        dist = manhattan_distance(goose_x, goose_y, food_x, food_y)
        closest_food = min(closest_food, dist)

    new_closest_food = float("inf")
    for food_pos in new_state.food:
        food_x, food_y = row_col(food_pos, NUM_COLS)
        dist = manhattan_distance(
            new_goose_x, new_goose_y, food_x, food_y)
        new_closest_food = min(closest_food, dist)

    # print(new_closest_food, closest_food)
    # if we move closer to food then give a little reward
    # note we could have just gotten lucky with a food spawn
    if new_closest_food <= closest_food:
        return (18 - new_closest_food)**2

    # gives smaller reward if food was close
    return - (18 - new_closest_food)


def train_single_episode(ddqn: DDQN):
    """
    Trains single episode
    :returns: total reward for the episode and whether or not our goose was the victor
    """
    total_episode_reward = 0
    step_counter = 0
    done = False

    state = trainer.reset()
    state_vector = create_norm_state_vector(state, None)

    while not done or step_counter == 200:
        action_vector = ddqn.choose_action(state_vector, state)
        action = get_action(action_vector)

        # Take our action
        new_state, reward, done, _ = trainer.step(action)
        new_state_vector = create_norm_state_vector(new_state, state)

        # reward = reward + calculate_reward(state, new_state)
        if done and reward == 0:
            reward = -1000

        # Store for experience replay
        ddqn.store_transition(
            state_vector,
            action_vector,
            reward,
            new_state_vector, done)

        ddqn.learn()

        step_counter += 1
        state = new_state
        state_vector = new_state_vector
        total_episode_reward += reward

    return total_episode_reward, len(new_state.geese[0]) != 0


def train(ddqn: DDQN, num_episodes, opt):
    rewards = []
    wins = []
    net_name = opt.net_type

    for i in trange(num_episodes, miniters=num_episodes / 500):
        ddqn.ep_decay(num_episodes, i)
        reward, did_win = train_single_episode(ddqn)
        if i > 50:
            ddqn.writer.add_scalar('average episode reward', np.mean(rewards[-50:]), i)
            ddqn.writer.add_scalar('average_wins', np.mean(wins[-50:]), i)

        if i % 500 == 0:
            ddqn.writer.flush()
            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)

        rewards.append(reward)
        wins.append(did_win)

        # Save model, rewards, and wins every save interval
        if i % ddqn.opt.save_interval == 0:
            ddqn_path = os.path.join(
                ddqn.opt.save_path,
                f"ddqn-{net_name}-{str(i).zfill(5)}.pt")
            # reward_path = os.path.join(ddqn.opt.save_path, 'rewards-' + str(i).zfill(5))
            # wins_path = os.path.join(ddqn.opt.save_path, 'wins-' + str(i).zfill(5))
            print(
                "saving model at episode %i in save_path=%s" %
                (i, ddqn_path))
            ddqn.save(ddqn_path)
            # np.savetxt(reward_path, rewards, fmt="%4.1d")
            # np.savetxt(wins_path, wins, fmt="%1d")

    # save final info
    ddqn_path = os.path.join(
        ddqn.opt.save_path,
        f"ddqn-{net_name}-final.pt")
    # reward_path = os.path.join(ddqn.opt.save_path, 'rewards-final-' + str(num_episodes).zfill(5))
    # wins_path = os.path.join(ddqn.opt.save_path, 'wins-final-' + str(num_episodes).zfill(5))
    print("saving final model")
    ddqn.save(ddqn_path)
    # np.savetxt(reward_path, rewards, fmt="%4.1d")
    # np.savetxt(wins_path, wins, fmt="%1d")
    print("Finished Training")

import sys
import numpy as np
from graphics import render_env
from model import DDQN
from kaggle_environments import make
from tqdm import trange
# from agent import agent
from vector import get_action, create_norm_state_vector
from typing import Tuple

env = make("hungry_geese")
trainer = env.train([None, "greedy", "greedy", "greedy"])


def test_single_episode(render=False) -> Tuple[bool, int]:
    # observations = env.run([agent, "greedy", "greedy", "greedy"])

    if render:
        render_env(env)
        for i, o in enumerate(observations):
            print("Iteration :", i)
            for observed in o:
                print(observed)
            print()

        response = input(
            "Press any key to continue or q to stop the program:\n")
        print("Player 0 is the white goose")
        if response == "q":
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


def test_single_episode_with_trainer(
        ddqn: DDQN, render: bool) -> Tuple[bool, int]:
    observations = []
    action_vectors = []
    actions = []
    state = trainer.reset()
    state_vector = create_norm_state_vector(state, None)

    observations.append(state)

    done = False
    while not done:
        action_vector = ddqn.choose_action(state_vector, state)
        action_vectors.append(action_vector)
        action = get_action(action_vector)
        actions.append(action)
        new_state, _, done, _ = trainer.step(action)
        state = new_state
        state_vector = create_norm_state_vector(new_state, state)
        observations.append(new_state)

    if render:
        # render_env(env)
        for i, o in enumerate(observations):
            print("Iteration :", i)
            print(o)
            if i < len(action_vectors):
                print(action_vectors[i])
                print(actions[i])
            print()

        response = input(
            "Press any key to continue or q to stop the program:\n")
        print("Player 0 is the white goose")
        if response == "q":
            quit()

    final_player_goose = new_state.geese[0]
    return len(final_player_goose) > 0, len(final_player_goose)


def test(ddqn: DDQN, num_episodes: int, render=False) -> None:
    num_wins = 0
    goose_lengths = []
    for i in trange(num_episodes):
        was_win, player_goose_length = test_single_episode_with_trainer(
            ddqn, render)

        if was_win:
            num_wins += 1

        goose_lengths.append(player_goose_length)

    s = "Num Wins/Num Episodes: {}/{} {}\nAverage Goose Length: {}\nMin Goose Length: {}\nMax Goose Length: {}\n"
    print(
        s.format(
            num_wins,
            num_episodes,
            num_wins /
            num_episodes,
            np.mean(goose_lengths),
            min(goose_lengths),
            max(goose_lengths)))

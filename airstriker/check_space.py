import gym
import sys
import retro
from gym.spaces import *

ENV_ID = sys.argv[1]

def print_spaces(label, space):
    print(label, space)

    if isinstance(space, Box):
        print(f"min: {space.low}")
        print(f"max: {space.high}")
    if isinstance(space, Discrete):
        print(f"min: {0}")
        print(f"max: {space.n-1}")

env = retro.make(game=ENV_ID)
print_spaces(f"行動空間: {env.action_space}")
print_spaces(f"状態空間: {env.observation_space}")
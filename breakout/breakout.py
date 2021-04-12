import gym
import time
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from util import callback, log_dir

ENV_ID = "BreakoutNoFrameskip-v0"
NUM_ENV = 8

def make_env(env_id)
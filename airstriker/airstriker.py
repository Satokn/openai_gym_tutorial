import retro
import time
import os
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor
from baselines.common.retro_wrappers import *
from util import log_dir, SaveOnBestTrainingRewardCallback, AirstrikerDiscretizer, CustomRewardAndDoneEnv

env = retro.make(game="Airstriker-Genesis", state="Level1")
env = AirstrikerDiscretizer(env)
env = CustomRewardAndDoneEnv(env)
env = StochasticFrameSkip(env, n=4, stickprob=0.25)
env = Downsample(env, 2)
env = Rgb2gray(env)
env = FrameStack(env, 4)
env = ScaledFloatFrame(env)
env = Monitor(env, log_dir, allow_early_resets=True)
print(f"行動空間: {env.action_space}")
print(f"状態空間: {env.observation_space}")

env.seed(0)
set_global_seeds(0)

env = DummyVecEnv([lambda: env])

model = PPO2("CnnPolicy", env, verbose=0)

model = PPO2.load("./logs/best_model.zip", env=env, verbose=0)

#model.learn(total_timesteps=128000, callback=SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir))

state = env.reset()
total_reward = 0
while True:
    env.render()
    time.sleep(1/60)
    action, _ = model.predict(state)

    state, reward, done, info = env.step(action)
    total_reward += reward[0] 
    
    if done:
        print(f"reward: {total_reward}")
        state = env.reset()
        total_reward = 0
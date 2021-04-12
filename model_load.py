import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make("CartPole-v1")
env = DummyVecEnv([lambda: env])

model = PPO2.load("sample")

state = env.reset()

for i in range(200):
   env.render()
   action, _ = model.predict(state, deterministic=True)
   state, reward, done, info = env.step(action)
   if done:
      break

env.close()
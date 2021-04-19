import retro
import time

env = retro.make(game="Airstriker-Genesis", state="Level1")

state = env.reset()
while True:
    env.render()

    time.sleep(1/60)
    state, reward, done, info = env.step(env.action_space.sample())

    if done:
        print("done")
        state = env.reset()
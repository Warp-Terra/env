import time
from box_world import Environment_Decision as BoxWorld

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env = BoxWorld(isGUI=True,is_save_images=True)
env.reset()
env.seed(1)

step = 0
total_reward = 0
done = False
while True:
    print("*****")
    x = input()
    try:
        x = int(x)
    except:
        print("not an integral, try again.")
    state, reward, done, info = env.step((x))
    step += 1
    total_reward += reward
    print(env.check_stuck_list)
    print("state", state)
    print(f"Step: {step}\nReward: {reward}\nTotal Reward: {total_reward}\ndone: {done}\ninfo: {info}")
    if done:
        print("*"*20, "Reset, New Game", "*"*20)
        env.reset()
        step = 0
        total_reward = 0
        done = False

exit()
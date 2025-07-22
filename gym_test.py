import logging, log_setting

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(log_setting.MyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import matplotlib.pyplot as plt
import time, os

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')

env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True

for step in range(10000):
    if done:
        state = env.reset()
    state, reward, done, trunc, info = env.step(env.action_space.sample())
    env.render()
    # plt.imshow(state)
    # plt.pause(0.00001)
    # logger.debug(env.render())
    # os.system("pause")
    # logger.debug(f"step: {step}, state: {state}, reward: {reward}, done: {done}, trunc: {trunc}, info: {info}")

# env.close()



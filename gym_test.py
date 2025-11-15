# log settings
import logging, log_setting

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(log_setting.MyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import matplotlib.pyplot as plt
import skimage
import time, os
import numpy as np
from typing import Tuple

import torch

# import mario libs
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import copy


env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
env = JoypadSpace(env, COMPLEX_MOVEMENT)


done = False

logger.debug(env.get_action_meanings())

logger.debug(env.reset()[0])
logger.debug(env.reset()[1])

for episode_num in range(5):
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0

    episode_over = False

    with open ("log.txt", "a") as f:
        print("-"*170, file=f)

    # for every episode:
    while not done:
        # Sample a random action from the environment's action space
        action = env.action_space.sample()  # Agent's policy can replace this for specific decision-making logic
 
        # Step the environment using the sampled action
        observation, reward, done, _, info = env.step(action)
        last_obs = copy.deepcopy(observation)

        state, reward, done, _, info = env.step(env.action_space.sample())
        env.render()

        with open ("log.txt", "a") as f:
            print(f"step {step_count}: \n    reward: {reward}, \n    done: {done}, \n    info: {info}, \n    {time.strftime('%Y-%m-%d %H:%M:%S')}", file=f)

        logger.debug(f"step: {step_count}, state: {state}, reward: {reward}, done: {done}, info: {info}")

env.close()


def downscale_obs(obs: np.ndarray, new_size: Tuple=(240, 256), to_gray: bool=True) -> np.ndarray:
    gray = np.dot(obs[:,:,0:3], [0.2989, 0.5870, 0.1140]) / 256
    # gray = skimage.color.rgb2gray(obs)
    # logger.debug(gray)

    return gray if to_gray else obs

# plt.imshow(env.render())
# plt.show()
# plt.imshow(downscale_obs(env.render()), cmap='gray')
# plt.show()
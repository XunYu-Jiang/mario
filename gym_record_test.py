# log settings
import logging, log_setting

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(log_setting.MyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import numpy as np
import copy

# import mario libs
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import gymnasium as gym
from gym.utils.save_video import save_video

# create mario env
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="rgb_array")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

# env = gym.make('CartPole-v1', render_mode="rgb_array_list")

# env = gym.wrappers.RecordVideo(env=env, video_folder="./temp", name_prefix="test", episode_trigger=lambda x: True)
logger.debug(env.metadata)

step_index = 0

for i in range(5):
    termination, truncation = False, False
    obs, info = env.reset()
    record_frames = []

    logger.debug(obs.shape)
    logger.warning(f"Episode {i} starting......")

    while not termination:
        state, reward, termination, truncation, info = env.step(env.action_space.sample())
        record_frames.append(state.copy())

        step_index += 1

    # for arr in record_frames:
    #     logger.debug(np.array_equiv(arr, state))

    logger.warning(f"episode {i} ended")
    logger.debug(np.shape(record_frames))
    logger.warning("Saving video......")
    save_video(
        frames=record_frames,
        video_folder="./temp",
        name_prefix="test",
        fps=env.metadata["video.frames_per_second"],
        episode_trigger=lambda x: True,
        # step_trigger=lambda x : True,
        # step_starting_index=step_starting_index,
        episode_index=i
    )
    logger.debug((termination, truncation))

env.close()
# logger.debug((termination, truncation, info))
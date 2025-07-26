# log settings
import logging, log_setting

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(log_setting.MyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import numpy as np
from typing import Tuple
import torch

# import mario libs
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

def train():
    """
    main train process
    """
    return

def downscale_obs(obs: np.ndarray, new_size: Tuple=(240, 256), to_gray: bool=True) -> np.ndarray:
    """
    downscale observation to grayscale.

    Args:
        obs (np.ndarray): observation of game state with ndarray and shape of (240, 256, 3).
        new_size (Tuple): new size of observation.
        to_gray (bool): whether to convert observation to grayscale.
    
    Returns:
        np.ndarray: downscaled grayscale observation with ndarray and shape of (240, 256).
    """
    gray = np.dot(obs[:,:,0:3], [0.2989, 0.5870, 0.1140]) / 256
    # gray = skimage.color.rgb2gray(obs)

    return gray if to_gray else obs

def prepare_state(state: np.ndarray, add_batch_dim: bool=True) -> torch.Tensor:
    """
    downscale one state of observation to grascale and turn it from ndarray to tensor. add batch dimension(optional). (240, 256, 3) -> (240, 256) -> (1, 240, 256)

    Args:
        state (np.ndarray): A np.ndarray state observation with shape of (240, 256, 3).
    
    Returns:
        A torch.Tensor which is a downscaled grayscale state observation with shape of (1, 240, 256).
    """
    if add_batch_dim:
        new_state = torch.from_numpy(downscale_obs(state)).to(dtype=torch.float32).unsqueeze(dim=0)
    else:
        new_state = torch.from_numpy(downscale_obs(state)).to(dtype=torch.float32)

    return new_state

def prepare_initial_state(init_state: np.ndarray) -> torch.Tensor:
    """
    get initial state, copy three times and add batch dimension. (240, 256, 3) -> (1, 3, 240, 256)

    Args:
        init_state (torch.Tensor): A np.ndarray state observation with shape of (240, 256, 3).
    
    Returns:
        A torch.Tensor which is a downscaled grayscale state observation with (batch, 3 observations, height, width).
    """
    gray_state = prepare_state(init_state, add_batch_dim=False)
    duplicate_states = gray_state.repeat(3, 1, 1)

    return duplicate_states.unsqueeze(dim=0)


def prepare_multi_state(states_queue: torch.Tensor, new_state: np.ndarray) -> torch.Tensor:
    """
    get past three states and new observation, discard the oldest one and add new observation to the end.

    Args: 
        states_queue (torch.Tensor): past three states with shape of (batch, past 3 states, height, width).
        new_state (np.ndarray): new state observation with shape of (height, width, channel).
    
    Returns:
        A torch.Tensor which is a downscaled grayscale state observation with (batch, 4 observations, height, width).
    """
    states_queue = torch.clone(states_queue)
    gray_state = prepare_state(new_state, add_batch_dim=False)

    states_queue[0][0] = states_queue[0][1]
    states_queue[0][1] = states_queue[0][2]
    states_queue[0][2] = gray_state

    return states_queue


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env.reset()
    logger.debug(np.shape(env.render()))
    temp = prepare_initial_state(env.render())
    logger.debug(temp.shape)
    temp2 = prepare_multi_state(temp, env.render())
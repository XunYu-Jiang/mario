# log settings
import log_setting

logger = log_setting.MyLogging.get_default_logger()

import numpy as np
from typing import Tuple, Callable, List, Dict
import torch
import time
from policy import Policy
from args import Args
import os
from rich.progress import track
from experience_replay import ExperienceReplay

# testing
from nnet_wrapper import NNetWrapper
from adv_actor_critic_nnet import AdvActorCriticNNet
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gymnasium as gym

class Coach():
    def __init__(self, env: gym.Env, nnet: NNetWrapper, policy: Callable) -> None:
        # game = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
        # self.env = JoypadSpace(game, COMPLEX_MOVEMENT)
        self._env = env
        self._env.reset()
        self._nnet = nnet
        self._policy = policy
        self._ex_replay = ExperienceReplay(batch_size=Args.NN_ARGS["batch_size"], buffer_size=Args.COACH_ARGS["buffer_size"])
        self.ACTION_NAMES = self._env.get_action_meanings()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"env action space: {self._env.action_space}")

        self._log_path = os.path.join(Args.FILE_ARGS["log_dir"], Args.FILE_ARGS["log_file"])

        if not os.path.exists(Args.FILE_ARGS["log_dir"]):
            os.mkdir(Args.FILE_ARGS["log_dir"])
        

    def _downscale_obs(self, obs: np.ndarray, new_size: Tuple=(240, 256), to_gray: bool=True) -> np.ndarray:
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


    def _prepare_state(self, state: np.ndarray, add_batch_dim: bool=False) -> torch.Tensor:
        """
        downscale one state of observation to grascale and turn it from ndarray to tensor. add batch dimension(optional). (240, 256, 3) -> (240, 256) -> (1, 240, 256)

        Args:
            state (np.ndarray): A np.ndarray state observation with shape of (240, 256, 3).
        
        Returns:
            A torch.Tensor which is a downscaled grayscale state observation with shape of (1, 240, 256) if add_batch_dim else (240,256).
        """
        if add_batch_dim:
            new_state = torch.from_numpy(self._downscale_obs(state)).to(dtype=torch.float32).unsqueeze(dim=0)
        else:
            new_state = torch.from_numpy(self._downscale_obs(state)).to(dtype=torch.float32)

        return new_state


    def _prepare_initial_state(self, init_state: np.ndarray, add_batch_dim: bool=False) -> torch.Tensor:
        """
        get original state, downscale it, turn it from ndarray to tensor,copy three times and add batch dimension if needed. (240, 256, 3) -> (3, 240, 256) or (1, 3, 240, 256)

        Args:
            init_state (torch.Tensor): A np.ndarray state observation with shape of (240, 256, 3).
        
        Returns:
            A torch.Tensor which is a downscaled grayscale state observation with (batch, 3 observations, height, width) or (3 observations, height, width).
        """
        gray_state = self._prepare_state(init_state, add_batch_dim=False)
        duplicate_states = gray_state.repeat(3, 1, 1)

        return duplicate_states.unsqueeze(dim=0) if add_batch_dim else duplicate_states



    def _prepare_multi_state(self, states_queue: torch.Tensor, new_state: np.ndarray) -> torch.Tensor:
        """
        get past three states and new observation, discard the oldest one and add new observation to the end.

        Args: 
            states_queue (torch.Tensor): past three states with shape of (batch, past 3 states, height, width).
            new_state (np.ndarray): new state observation with shape of (height, width, channel).
        
        Returns:
            A torch.Tensor which is a downscaled grayscale state observation with (batch, 4 observations, height, width).
        """
        states_queue = states_queue.clone()
        new_gray_state = self._prepare_state(new_state, add_batch_dim=False)

        states_queue[0] = states_queue[1]
        states_queue[1] = states_queue[2]
        states_queue[2] = new_gray_state

        return states_queue


    def _caculate_cumulated_reward(self):
        """
        Caculate cumulated reward every state in one self_play
        """
        pass

    def _get_dataloader(self):
        """
        create dataloader from replay buffer
        """
        pass

    def _get_action(self, state_queue: torch.Tensor) -> int:
        # get policy and value from nnet
        # self._policy(self._nnet.predict(state_queue)[0])
        values_pred: torch.Tensor = torch.randint(0, 12, (1,))    # get qvalues from nnet

        return values_pred.item()

    def _log_self_play(self):
        pass

    def reset_env(self) -> None:
        """
        Reset the environment outside of Coach class.
        """
        self._env.reset()

    def _self_play(self) -> List[Tuple[torch.Tensor,int, float, Dict]]:
        """
        Perform one episode of self-play.

        Returns:
            List[Tuple[torch.Tensor, int, float, Dict]]: A list of self-play records with format of (state_queue, action_index, reward, info).
        """
        # check if need qvalues in replay buffer later
        self_play_examples = []

        state, _ = self._env.reset()
        done: bool = False
        step_count: int = 0

        # downscale state to grayscale and turn it into format of nnet input (240, 256, 3) -> (240, 256) -> (3, 240, 256)
        states_queue: torch.Tensor = self._prepare_initial_state(state, add_batch_dim=False)

        # start logging self-play imformation
        with open (self._log_path, "a") as f:
            print("-"*150, file=f)

        while not done:
            action_index: int = self._get_action(states_queue)
            # logger.debug(self.ACTION_NAMES[action_index])

            state, reward, done, _, info = self._env.step(action_index)
            step_count += 1
            self._env.render()

            # get a new tensor every time
            states_queue = self._prepare_multi_state(states_queue, state)
            # logger.debug((states_queue.shape, reward, done, info))

            with open(self._log_path, "a") as f:
                print(f"step {step_count} at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n    reward: {reward}, \n    info: {info}", file=f)
            
            # need to check if state queue overwrited each state
            self_play_examples.append((states_queue, action_index, reward, info))
        
        logger.debug(len(self_play_examples))

        return self_play_examples


    def learn(self):
        """
        main training process
        """

        num_iters: int = Args.COACH_ARGS["num_iters"]
        num_episodes: int = Args.COACH_ARGS["num_episodes"]

        for iter in track(range(num_iters), description="[blue]Start: Iteration"):
            for episode in track(range(num_episodes), description="[green]Start: Self Play", transient=True):
                episode_example = self._self_play()
                self.ex_replay.add_replay(episode_example)
                # time.sleep(0.5)



def test_process_state():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    coach = Coach(env=env, nnet=NNetWrapper(AdvActorCriticNNet()), policy=Policy.episilon_greedy)

    state, _ = env.reset()

    #original state
    reset_state = state.copy()
    logger.debug(f"reset state: {reset_state.shape}, {type(reset_state)}")

    #check coach._downscale_obs()
    downscaled_state = state.copy()
    downscaled_state = coach._downscale_obs(downscaled_state)
    logger.debug(f"downscale_obs: {downscaled_state.shape}, {type(downscaled_state)}")

    #check coach._prepare_state()
    prepared_state = state.copy()
    prepared_state = coach._prepare_state(prepared_state, add_batch_dim=True)
    logger.debug(f"prepare_state add batch: {prepared_state.shape}, {type(prepared_state)}")

    prepared_state = state.copy()
    prepared_state = coach._prepare_state(prepared_state, add_batch_dim=False)
    logger.debug(f"prepare_state: {prepared_state.shape}, {type(prepared_state)}")

    #check coach.prepare_initial_state()
    init_state = state.copy()
    init_state = coach._prepare_initial_state(init_state, add_batch_dim=True)
    logger.debug(f"prepare_initial_state add batch: {init_state.shape}, {type(init_state)}")

    init_state = state.copy()
    init_state = coach._prepare_initial_state(init_state, add_batch_dim=False)
    logger.debug(f"prepare_initial_state: {init_state.shape}, {type(init_state)}")
    logger.debug(f"check if three states are equal: {torch.equal(init_state[0], init_state[1])}, {torch.equal(init_state[1], init_state[2])}")
    
    # check coach._prepare_multi_state()
    state, reward, done, _, info = env.step(env.action_space.sample())

    multi_state = init_state.detach()
    multi_state = coach._prepare_multi_state(multi_state, state)
    logger.debug(f"prepare_multi_state: {multi_state.shape}, {type(multi_state)}")
    logger.debug(f"check if three states are equal: {torch.equal(multi_state[0], multi_state[1])}, {torch.equal(multi_state[1], multi_state[2])}")

    multi_state = init_state.detach()
    step_count = 0
    while not done:
        state, reward, done, _, info = env.step(env.action_space.sample())
        multi_state = coach._prepare_multi_state(multi_state, state)
        logger.debug(f"check if three states are equal: {torch.equal(multi_state[0], multi_state[1])}, {torch.equal(multi_state[1], multi_state[2])}")

        step_count += 1
        if step_count > 30:
            break

def main():
    logger.debug(logger.name)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    coach = Coach(env=env, nnet=NNetWrapper(AdvActorCriticNNet()), policy=Policy.episilon_greedy)

    coach.reset_env()
    # coach._self_play()
    coach.learn()

if __name__ == "__main__":
    # test_process_state()
    main()

    
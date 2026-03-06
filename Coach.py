# log settings
import log_setting

logger = log_setting.MyLogging.get_root_logger()

import numpy as np
from typing import Tuple, Callable, List, Dict
import torch
import time
from algorithm import Algorithom
from args import Args
import os
import copy

from experience_replay import ExperienceReplay, CustomDataSet
from pathlib import Path

# testing
from nnet_wrapper import NNetWrapper
from q_nnet import Q_network
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import gymnasium as gym
from gym.utils.save_video import save_video

from tqdm import trange
from torch.utils.data import DataLoader


# Use tqdm.write instead of print to avoid progress bar breaks

class Coach():
    def __init__(self, env: gym.Env, nnet: NNetWrapper, policy: Callable) -> None:
        self._env = env
        self._env.reset()
        self._nnet = nnet
        self._policy = policy
        self.ex_replay = ExperienceReplay(batch_size=Args.TRAIN_ARGS["batch_size"], buffer_size=Args.COACH_ARGS["buffer_size"])
        self._replay_buffer = self.ex_replay.get_replay_buffer()
        self.ACTION_NAMES = self._env.get_action_meanings()
        # self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DEVICE = Args.TRAIN_ARGS["device"]
        logger.info(f"env action space: {self._env.action_space}")

        self._log_path = os.path.join(Args.FILE_ARGS["log_dir"], Args.FILE_ARGS["log_file"])
        self.num_iters: int = Args.COACH_ARGS["num_iters"]
        self.num_episodes: int = Args.COACH_ARGS["num_episodes"]

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
        state = np.array(state).astype(np.float32)
        if add_batch_dim:
            new_state = torch.from_numpy(self._downscale_obs(state)).unsqueeze(dim=0)
        else:
            new_state = torch.from_numpy(self._downscale_obs(state))

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


    def _get_cumulated_reward(self, states_queue: torch.Tensor) -> torch.Tensor:
        """
        Using self.nnet to caculate cumulated reward for every state in one episode of self_play. Append next_state_value to the end of every example.

        Args:
            self_play_examples (List[Tuple[torch.Tensor, int, float, Dict]]): one episode of self_play.
        """

        logger.warning("prediction not implemented")
        value_next_pred, _ = self._nnet.predict(states_queue)

        raise NotImplementedError
        return value_next_pred
        

    def _get_dataloader(self):
        """
        create dataloader from replay buffer
        """
        raise NotImplementedError

    def _get_action(self, value_pred: torch.Tensor) -> int:
        # get policy and value from nnet
        # tmp = torch.rand(12)
        # logger.debug(f"{value_pred.shape}")
        # logger.debug(value_pred)
        # logger.debug(f"{value_pred.argmax(dim=1)}")
        # logger.debug(f"{value_pred[value_pred.argmax(dim=1)]}")

        return value_pred.argmax(dim=1).item()

    def reset_env(self) -> None:
        """
        Reset the environment outside of Coach class, since self._env is not accessible outside of Coach class.
        """
        self._env.reset()

    # def record_play(self, record_frames: List[np.ndarray], record_dir: Path | str):

    def _self_play(self) -> Tuple[List[Tuple[torch.Tensor,int, float, Dict]], List[np.ndarray]]:
        """
        Perform one episode of self-play.
        self_play_example: A list of a single self-play records with format of 
                            (states_queue, action_index, reward, last_value_pred, value_pred, prob_pred, info).
        record_frames (List[np.ndarray]): A list of RAW RGB data of one episode of self-play.

        Returns:
            self_play_example (List[Tuple[torch.Tensor, int, float, Dict]]): A list of self-play records with format of (states_queue, action_index, reward, info).
        """

        #initialize data
        last_reward = 0.
        last_value_pred = torch.zeros(12)
        
        self_play_example = []
        record_frames = []
        
        state, _ = self._env.reset()
        done: bool = False
        step_count: int = 0

        record_frames.append(state.copy())
        # downscale state to grayscale and turn it into format of nnet input (240, 256, 3) -> (240, 256) -> (3, 240, 256)
        states_queue: torch.Tensor = self._prepare_initial_state(state.copy(), add_batch_dim=False)
        last_states_queue: torch.Tensor = states_queue.clone().detach()


        # start logging self-play imformation
        with open (self._log_path, "a") as f:
            print("-"*150, file=f)

        while not done:
            step_count += 1
            # prob_pred: torch.Tensor = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            with torch.inference_mode():

                # states_queue = states_queue.to(dtype=torch.float32, device=self.DEVICE)

                value_pred = self._nnet.predict(states_queue.unsqueeze(0).to(dtype=torch.float32, device=self.DEVICE))
                action_index: int = self._get_action(value_pred)

                value_pred = value_pred.squeeze(0).to(dtype=torch.float32, device="cpu")

                # logger.debug(f"{value_pred.shape}, {states_queue.shape}")
                # logger.debug(f"{value_pred.device}, {states_queue.device}")

                # interact with env
                state, reward, done, _, info = self._env.step(action_index)
                
                # add new state to record_frames
                record_frames.append(state.copy())
                
                # get a new copy of tensor_queue every time
                states_queue = self._prepare_multi_state(states_queue, state.copy())
                self._env.render()

                with open(self._log_path, "a") as f:
                    print(f"step {step_count} at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n    reward: {last_reward},\n    last_value_pred: {last_value_pred},\n    value_pred: {value_pred}\n    info: {info}", file=f)
                
                self_play_example.append((last_states_queue, last_reward, last_value_pred.clone()))  #(Tensor, float, Tensor)
                
                last_value_pred = value_pred
                last_states_queue = states_queue
                last_reward = reward

                if step_count > 30:
                    break

        # add last state to self_play_example with next_state_value = 0(since it is terminal state)
        self_play_example.append((last_states_queue, last_reward, last_value_pred))
        logger.debug(f"game length: {len(self_play_example)}")

        return self_play_example, record_frames

    def learn(self):
        """
        main training process
        """

        
        # episode_example, record_frames = self._self_play()
        # logger.debug((record_frames[0].shape, len(record_frames)))
        
        logger.warning("Start self play...")

        bar_fmt = "{desc}: |{bar}| {n_fmt}/{total_fmt} {remaining},{rate_fmt}{postfix}"

        # for every iteration
        for iteration in trange(self.num_iters,  desc="Iteration", colour="blue", bar_format=bar_fmt):
            # for every self-play
            for episode in trange(self.num_episodes, desc="Self Play", leave=False, colour="cyan", bar_format=bar_fmt):
                
                self_play_example, record_frames = self._self_play()
                # one self play ended, save video and records to experience replay

                save_video(
                    frames=record_frames,
                    video_folder="./temp",
                    name_prefix="test",
                    fps=self._env.metadata["video.frames_per_second"],
                    episode_trigger=lambda x: True,
                    # step_trigger=lambda x : True,
                    # step_starting_index=step_starting_index,
                    episode_index=f"{iteration + 1}-{episode + 1}"
                )

                self.ex_replay.add_replay(self_play_example)
            
            # get replay into dataset and dataloader...
            
            logger.warning(f"Start training iter {iteration + 1}...")
            custom_dataset = CustomDataSet(self._replay_buffer)
            train_dataloader = DataLoader(custom_dataset, batch_size=Args.TRAIN_ARGS["batch_size"], shuffle=True, drop_last=True)

            X, y = next(iter(train_dataloader))
            logger.debug(f"{X.is_cuda}, {y[0].is_cuda}, {y[1].is_cuda}")
            self._nnet.train(dataloader=train_dataloader, device=self.DEVICE)
        
        logger.warning("Finish training")
    
                    




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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    coach = Coach(env=env, nnet=NNetWrapper((Q_network()), device=device), policy=Algorithom.Policy.episilon_greedy)

    coach.reset_env()
    coach.learn()
    # coach._self_play()

if __name__ == "__main__":
    # test_process_state()
    main()


    
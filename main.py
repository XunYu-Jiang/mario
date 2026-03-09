# logging
from log_setting import MyLogging
logger = MyLogging.get_root_logger()

# import mario libs
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# depandencies
from coach import Coach
from nnet_wrapper import NNetWrapper
from adv_actor_critic_nnet import AdvActorCriticNNet
from algorithm import Algorithom
from q_nnet import Q_network
from args import Args

import torch

import contextlib
import inspect
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

@contextlib.contextmanager
def print_redirect_tqdm():
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.write raises error, use built-in print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print

def main():

    with logging_redirect_tqdm(), print_redirect_tqdm():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)

        nnet = Q_network()
        optimizer = torch.optim.Adam(nnet.parameters(), lr=Args.TRAIN_ARGS["lr"])
        nnet_wrap = NNetWrapper(nnet=nnet, optimizer=optimizer, device=device)
        
        coach = Coach(env=env, nnet=nnet_wrap, policy=Algorithom.Policy.episilon_greedy)

        coach.reset_env()
        coach.learn()
    

if __name__ == "__main__":
    main()


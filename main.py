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
from policy import Policy

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
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        coach = Coach(env=env, nnet=NNetWrapper(AdvActorCriticNNet()), policy=Policy.episilon_greedy)
        coach.learn()
    

if __name__ == "__main__":
    main()


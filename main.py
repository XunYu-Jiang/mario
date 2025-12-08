# logging
from log_setting import MyLogging
logger = MyLogging.get_default_logger()

# import mario libs
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

# depandencies
from coach import Coach
from nnet_wrapper import NNetWrapper
from adv_actor_critic_nnet import AdvActorCriticNNet
from policy import Policy

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    coach = Coach(env=env, nnet=NNetWrapper(AdvActorCriticNNet()), policy=Policy.episilon_greedy)
    coach.learn()
    

if __name__ == "__main__":
    main()


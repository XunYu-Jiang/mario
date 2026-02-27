# log settings
import log_setting

import torch
import numpy as np
from typing import Tuple
from nnet_wrapper import NNetWrapper


logger = log_setting.MyLogging.get_root_logger()

class Policy:
    def __init__(self, value_function: NNetWrapper) -> None:
        self._nnet = value_function
        self._cumulated_reward: float = 0.

    # def _get_cumulated_reward(self, obs: torch.Tensor) -> torch.Tensor | Tuple:
    #     """
    #     get cumulated reward from nnet.
    #     """
    #     return self._nnet.predict()

    # @classmethod
    # def adv_actor_critic(cls, obs: np.ndarray):
    #     """
    #     advantageactor-critic policy. Get critic as well as cumulated reward from nnet.
    #     """
    #     pass
    def policy_gradient(self):
        raise NotImplementedError
    
    def get_action_by_prob(self, input: torch.Tensor, replacement: bool=True) -> int:
        """
        choose one action according to it's probability.

        Args:
            input (torch.Tensor): policy-head output(1-D tensor)..
            replacement (bool): whether to sample with replacement.
        """



    @staticmethod
    def episilon_greedy(input: torch.Tensor, eps: float=0.):
        """
        select the action according to the softmaxed qvalues in action space, and choose one action according to it's probability. (why normalized??????)

        Args:
            input (torch.Tensor): policy or value from nnet (1-D tensor).
            eps (float): epsilon for epsilon-greedy policy, if None, then return the action with the highest qvalue.
        
        Returns:
            A torch.Tensor which is an action in action space.
        """
        # using epsilon-greedy
        if eps != 0:
            if torch.rand(1) < eps:
                return torch.randint(low=0, high=7, size=(1,))
            else:
                return torch.argmax(input)
        
        # not using epsilon-greedy
        # choose qvalue action according to it's normalized qvalue probability
        return torch.multinomial(input, num_samples=1, replacement=True)
    
def main():
    pass

if __name__ == "__main__":
    main()
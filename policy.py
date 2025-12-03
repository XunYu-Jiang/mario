# log settings
import log_setting

logger = log_setting.MyLogging.get_default_logger()

import torch

class Policy:
    @staticmethod
    def adv_actor_critic(cumulated_reward_model: torch.Module):
        """
        advantageactor-critic policy. Get critic as well as cumulated reward from nnet.
        """
        pass


    @staticmethod
    def episilon_greedy(qvalues: torch.Tensor, eps: float=None):
        """
        select the action according to the softmaxed qvalues in action space, and choose one action according to it's probability. (why normalized??????)

        Args:
            qvalues (torch.Tensor): qvalues of each action in action space.
            eps (float): epsilon for epsilon-greedy policy, if None, then return the action with the highest qvalue.
        
        Returns:
            A torch.Tensor which is an action in action space.
        """
        # using epsilon-greedy
        if eps is not None:
            if torch.rand(1) < eps:
                return torch.randint(low=0, high=7, size=(1,))
            else:
                return torch.argmax(qvalues)
        
        # not using epsilon-greedy
        # choose qvalue action according to it's normalized qvalue probability
        return torch.multinomial(torch.functional.normalize(qvalues), num_samples=1)
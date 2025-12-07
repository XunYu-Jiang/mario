from typing import Callable
import torch

class Engine():
    """
    implement detail training, evaluation, etc behavior for NNetWrapper.
    """
    def __init__(self):
        pass 

    @classmethod
    def train(cls,
              nnet: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: Callable, 
              optimizer: torch.optim.Optimizer,
              device: torch.device) -> None:
        train_loss = 0

        # seudo code
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
        
            policy_pred,value_pred = nnet(X)
            action = policy_pred.sample()

            # think where to put
            advantage = (reward + gamma * next_value) - value_pred
            loss = -1 * policy_pred.logprob(action) * advantage
            min(loss)

        pass

    def evaluate(self):
        pass

    def get_loss_fn(self):
        pass

    def get_nnet_output(self):
        pass
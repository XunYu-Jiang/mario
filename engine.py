from typing import Callable
import torch

class Engine():
    """
    implement detail training, evaluation, etc behavior for NNetWrapper.
    """
    def __init__(self):
        pass 

    
    def train(cls,
              nnet: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: Callable, 
              optimizer: torch.optim.Optimizer,
              device: torch.device) -> None:
        train_loss = 0

        # seudo code
        for batch, X in enumerate(dataloader):
            optimizer.zero_grad()

            X = X.to(device)
            last_reward, last_value_pred, value_pred = X[0], X[1], X[2], X[3]

            # think where to put
            advantage = (last_reward + gamma * value_pred) - last_value_pred

            actor_loss = -1 * last_prob_pred.logprob(action) * advantage
            critic_loss = torch.pow(advantage, 2)
            total_loss = actor_loss.sum() + discount * critic_loss.sum()

            total_loss.backward()
            optimizer.step()

        pass
    
    def evaluate(self):
        pass

    def get_loss_fn(self):
        pass

    def get_nnet_output(self):
        pass
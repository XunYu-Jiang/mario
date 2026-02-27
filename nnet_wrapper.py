import torch
from typing import Tuple

import log_setting
logger = log_setting.MyLogging.get_root_logger()

import engine

class NNetWrapper():
    def __init__(self, nnet: torch.nn.Module, device: torch.device=torch.device("cpu")) -> None:
        self._nnet = nnet
        self.device = device
        self._engine = engine.Engine()

    def train(self, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn:torch.nn.Module, 
              optimizer: torch.optim.Optimizer):
        """"""
        self._engine.train()
    
    def predict(self, obs: torch.Tensor) -> torch.Tensor | Tuple:
        # v_pred, p_pred = self._nnet(obs)

        return torch.zeros(1), torch.zeros(12)
        
    
    def save_checkpoint(cls, folder, filename):
        ...

    def load_checkpoint(self, folder, filename):
        ...


def main():
    a = 10
    logger.debug(a)

if __name__ == "__main__":
    main()
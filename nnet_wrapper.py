import torch
from typing import Tuple

import log_setting
logger = log_setting.MyLogging.get_root_logger()

from engine import Engine

class NNetWrapper():
    def __init__(self, nnet: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device=torch.device("cpu")) -> None:
        self._nnet = nnet
        self.optimizer = optimizer
        self.device = device
        self._engine = Engine(nnet=self._nnet, optimizer=self.optimizer, device=self.device)

    def train(self, 
              dataloader: torch.utils.data.DataLoader,
              target: bool=True,
              target_nnet: torch.nn.Module=None
              ):
        """Wrapper of engine.train()"""

        self._engine.train(dataloader=dataloader, target=target, target_nnet=target_nnet)
    
    def predict(self, state_queue: torch.Tensor) -> torch.Tensor | Tuple:
        return self._engine.predict(state_queue)
        
    
    def save_checkpoint(self, folder, filename):
        ...

    def load_checkpoint(self, folder, filename):
        ...
    
    def get_nnet_instance(self):
        return self._nnet
                          
    def get_loss(self):
        return self._engine.get_loss()


def main():
    a = 10
    logger.debug(a)

if __name__ == "__main__":
    main()
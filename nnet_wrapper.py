import torch
from typing import Tuple

import log_setting
logger = log_setting.MyLogging.get_default_logger()


class NNetWrapper():
    def __init__(self, game, nnet: torch.nn.Module):
        raise NotImplementedError

    def train(self, train_data: torch.Tensor):
        
        raise NotImplementedError
    
    def predict(self, obs: torch.Tensor) -> torch.Tensor | Tuple:
        ...
    
    def save_checkpoint(cls, folder, filename):
        ...

    def load_checkpoint(self, folder, filename):
        ...

def main():
    a = 10
    logger.debug(a)

if __name__ == "__main__":
    main()
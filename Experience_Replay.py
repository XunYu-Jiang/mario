# log settings
import logging, log_setting

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(log_setting.MyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from random import shuffle
import torch
import numpy as np
from collections import deque
from typing import Tuple

class ExperienceReplay:
    def __init__(self, N, buffer_size=1000) -> None:
        self.N = N
        self.replay_buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)

    def add_replay(self, state1, action, reward, state2) -> None:
        
        if len(self.replay_buffer) < self.replay_buffer_size:
            self.replay_buffer.append((state1, action, reward, state2))
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append((state1, action, reward, state2))
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        get replays from replay buffer and encapsulate them into batches like
        """
        pass

if __name__ == "__main__":
    dq = deque(maxlen=5)
    logger.debug(dq)

    for i in range(10):
        dq.append(i)
        logger.debug(dq)

    dq.popleft()
    logger.debug(dq)
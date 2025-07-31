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
    def __init__(self, batch_size=32, buffer_size=100) -> None:
        self.batch_size = batch_size
        self.replay_buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)

    def add_replay(self, states, action, reward, info) -> None:
        """
        add grayscale states, action, reward, info into replay buffer.

        Args:
            states (torch.Tensor): last 3 grayscale state observations. (3, 240, 256)
            action (int): last action from actor.
            reward (torch.Tensor): reward from mario env depands on last action.
            info (): last info from mario env.
        
        Returns:
            None
        """
        if len(self.replay_buffer) < self.replay_buffer_size:
            self.replay_buffer.append((states, action, reward, info))
        else:
            self.replay_buffer.popleft()
            self.replay_buffer.append((states, action, reward, info))
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        get replays from replay buffer and encapsulate them into batches like.
        """
        if len(self.replay_buffer) < self.batch_size:
            batch_size = len(self.replay_buffer)
        else:
            batch_size = self.batch_size
        
        if len(self.replay_buffer) < 1:
            logger.critical("No replay in buffer.")
            return None

        states_batch = torch.stack([replay[0] for replay in self.replay_buffer], dim=0)
        action_batch = torch.tensor([replay[1] for replay in self.replay_buffer]).to(dtype=torch.int64)
        reward_batch = torch.tensor([replay[2] for replay in self.replay_buffer])
        
        return states_batch, action_batch, reward_batch

if __name__ == "__main__":
    dq = deque(maxlen=100)
    logger.debug(dq)

    for i in range(10):
        dq.append(i)
        logger.debug(dq)

    dq.popleft()
    logger.debug(dq)
    logger.debug(len(dq))
    temp = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
    temp2 = torch.tensor([[7, 8, 9],
                          [10, 11, 12]])
    # logger.debug(f"{temp}, {temp.shape}")
    states_batch = torch.stack((temp, temp2), dim=0)
    # logger.debug(f"{states_batch}, {states_batch.shape}")

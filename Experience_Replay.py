# log settings
import log_setting

logger = log_setting.MyLogging.get_root_logger()

from random import shuffle
import torch
import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import copy
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards, infos, next_state_values):
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._infos = infos
        self._next_state_values = next_state_values

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        return self._states[idx], self._actions[idx], self._rewards[idx], self._infos[idx], self._next_state_values[idx]
class ExperienceReplay:
    def __init__(self, batch_size=32, buffer_size=100) -> None:
        self._batch_size = batch_size
        self._replay_buffer_size = buffer_size
        self._replay_buffer = deque(maxlen=buffer_size)

    def get_replay_buffer(self) -> deque:
        return self._replay_buffer
    
    def add_replay(self, states: torch.Tensor, action: int, reward: float, info: Dict, next_state_value: float) -> None:
        """
        add grayscale state_queue, action, reward, info into replay buffer.

        Args:
            states (torch.Tensor): last 3 grayscale state observations. (3, 240, 256)
            action (int): last action from actor.
            reward (torch.Tensor): reward from mario env depands on last action.
            info (dict): last info from mario env.
        
        Returns:
            None
        """
        if len(self._replay_buffer) < self._replay_buffer_size:
            self._replay_buffer.append((states, action, reward, info))
        else:
            self._replay_buffer.popleft()
            self._replay_buffer.append((states, action, reward, info))
    
    def get_last_replay(self) -> List[Tuple[torch.Tensor, int, float, Dict, float]]:
        """
        get a copy of last replay from replay buffer.

        Returns:
            List[torch.Tensor, int, float, dict]: last replay history
        """
        if len(self._replay_buffer) < 1:
            logger.critical("No replay in buffer.")
            return None
        
        return torch.clone(self._replay_buffer[-1][0]).detach(), self._replay_buffer[-1][1], self.replay_buffer[-1][2], copy.deepcopy(self.replay_buffer[-1][3])
    
    def get_batch(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        get replays from replay buffer and encapsulate them into batches like.
        """
        if len(self._replay_buffer) < self._batch_size:
            batch_size = len(self._replay_buffer)
        else:
            batch_size = self._batch_size
        
        if len(self._replay_buffer) < 1:
            logger.critical("No replay in buffer.")
            return None

        states_batch = torch.stack([replay[0] for replay in self._replay_buffer], dim=0)
        action_batch = torch.tensor([replay[1] for replay in self._replay_buffer]).to(dtype=torch.int64)
        reward_batch = torch.tensor([replay[2] for replay in self._replay_buffer])
        
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
    logger.debug(dq[-1])

    # logger.debug(f"{temp}, {temp.shape}")
    states_batch = torch.stack((temp, temp2), dim=0)
    # logger.debug(f"{states_batch}, {states_batch.shape}")
    dick = {"a": 1, "b": 2}
    cock = copy.deepcopy(dick)
    logger.debug((dick, id(dick)))
    logger.debug((cock, id(cock)))

    a = (1, 2)
    b = a[1]
    b = b + 1

    logger.debug((a, b))


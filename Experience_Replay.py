# log settings
import log_setting

logger = log_setting.MyLogging.get_root_logger()

from random import shuffle
import torch
import numpy as np
from collections import deque
from typing import List, Dict, Tuple
import copy
from args import Args

class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, replay_buffer: deque):
        # self.replay_buffer = replay_buffer
        self.state_queue = []
        self.reward = []
        self.value_pred = []
        
        for train_example in replay_buffer:
            for st, rw, vp in train_example:
                self.state_queue.append(st)
                self.reward.append(rw)
                self.value_pred.append(vp)

        # self.reward_stack = torch.tensor(self.reward, dtype=torch.float32, device=Args.TRAIN_ARGS["device"])
        

        
        # self.state_queue_stack = torch.tensor(self.state_queue, dtype=torch.float32, device=Args.TRAIN_ARGS["device"])
        # self.value_pred_stack = torch.tensor(self.value_pred, dtype=torch.float32, device=Args.TRAIN_ARGS["device"])
        # logger.debug(f"{self.state_queue_stack.shape}, {self.value_pred_stack.shape}")
        # logger.debug(self.reward_stack.shape)
        # logger.debug(f"{self.state_queue_stack.device}, {self.reward_stack.device}, {self.value_pred_stack.device}")

        ### need optimization

    def __len__(self):
        return len(self.reward)

    def __getitem__(self, idx):
        return self.state_queue[idx], (self.reward[idx], self.value_pred[idx])

class ExperienceReplay:
    def __init__(self, batch_size=32, buffer_size=1000) -> None:
        self._batch_size = Args.EX_REPLAY["batch_size"]
        self._replay_buffer_size = Args.EX_REPLAY["buffer_size"]
        self._replay_buffer: deque = deque(maxlen=buffer_size)

    def get_replay_buffer(self) -> deque:
        return self._replay_buffer
    
    def add_replay(self, self_play_example: Tuple) -> None:
        """
        add grayscale state_queue, action, reward, info into replay buffer.

        Args:
            self_play_example (Tuple): (last_state_queue, last_reward, last_value_pred, value_pred) in Q_learning
        
        Returns:
            None
        """
        if len(self._replay_buffer) < self._replay_buffer_size:
            self._replay_buffer.append(self_play_example)
        else:
            self._replay_buffer.popleft()
            self._replay_buffer.append(self_play_example)
    
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
def test_dataset():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = torch.stack((a, b), dim=0)
    logger.debug(c)
    d = [1, 2, 3]
    e = [a]
    f = torch.tensor(e)
    logger.debug(f)

def main():
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

if __name__ == "__main__":
    test_dataset()


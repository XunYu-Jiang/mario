from typing import Callable
from args import Args
import torch
from q_nnet import Q_network
import log_setting

logger = log_setting.MyLogging.get_root_logger()

class Engine():
    """
    implement detail training, evaluation, etc behavior for NNetWrapper.
    """
    def __init__(self, nnet: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device=torch.device("cpu")) -> None:
        self._nnet = nnet.to(Args.TRAIN_ARGS["device"])
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.device = device
        self.all_loses = []
    
    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        self._nnet.train()
        self.all_loses = []
        # seudo code
        for batch, (X, y) in enumerate(dataloader):

            self.optimizer.zero_grad()

            X = X.to(dtype=torch.float32, device=self.device)
            y[0] = y[0].to(dtype=torch.float32, device=self.device)
            y[1] = y[1].to(dtype=torch.float32, device=self.device)

            reward, last_value_pred = y[0].to(self.device), y[1].to(self.device)

            value_pred: torch.Tensor = self._nnet(X)

            # logger.debug(value_pred.requires_grad)
            # logger.debug(f"{value_pred.shape}, {reward.shape}, {last_value_pred.shape}")

            reward = reward.reshape(Args.TRAIN_ARGS['batch_size'], 1)

            # logger.debug(f"{value_pred.shape}, {reward.shape}, {last_value_pred.shape}")

            target = torch.add(reward, Args.TRAIN_ARGS['q_learning_discount'] * value_pred)

            loss: torch.Tensor = self.loss_fn(last_value_pred, target)
            
            # record all loses for tracking experiment
            tmp_loss = loss.clone().detach()
            tmp_loss = tmp_loss.sum()
            self.all_loses.append(tmp_loss.to(device="cpu"))

            loss.backward()
            self.optimizer.step()
    
    def predict(self, state_queue: torch.Tensor) -> torch.Tensor:
        return self._nnet(state_queue)

    def get_loss(self):
        return self.all_loses

def main():
    a = torch.randint(0, 10, size=(3,2))
    b = torch.randint(0, 10, size=(3,))
    logger.debug(f"a: {a}, {a.shape}")
    logger.debug(f"b: {b}, {b.shape}")
    b = b.reshape(3, 1)
    logger.debug(f"b: {b}, {b.shape}")
    tmp = []
    c = torch.add(a, b)
    logger.debug(f"c: {c}, {c.shape}")
    for i in range(3):
        c = torch.add(a, b[i])
        tmp.append(c)
    # for k in tmp:
    #     if len(tmp) == 0:
    #         break
    #     d = torch.cat(k, k+1, dim=1)
    #     logger.debug(f"c: {c}, {c.shape}")

if __name__ == "__main__":
    main()
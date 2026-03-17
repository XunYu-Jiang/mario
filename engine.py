from typing import Callable
from args import Args
import torch
from q_nnet import Q_network
import log_setting
import gc

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
        self.total_loses = 0.
    
    def train(self, dataloader: torch.utils.data.DataLoader, nnet: torch.nn.Module=None, target_nnet: torch.nn.Module=None) -> None:
        self.total_loses = 0.

        # if target_nnet is not None:
        #     nnet = target_nnet.to(device=self.device)
        # else:
        #     nnet = self._nnet.to(device=self.device)
        nnet = nnet.to(device=Args.TRAIN_ARGS["device"])
        target_nnet = target_nnet.to(device=Args.TRAIN_ARGS["device"])
        
        nnet.train()

        # seudo code
        for batch, (X, y) in enumerate(dataloader):
            # logger.warning(f"start batch {batch+1}")
            # logger.debug(f"{X.shape}, {y[0].shape}, {y[1].shape}")
            self.optimizer.zero_grad()

            X = X.to(dtype=torch.float32, device=self.device)
            reward, next_state = y[0].to(dtype=torch.float32, device=self.device), y[1].to(dtype=torch.float32, device=self.device)


            value_pred: torch.Tensor = nnet(X)
            next_value_pred: torch.Tensor = target_nnet(next_state)

            # logger.debug(value_pred.requires_grad)
            # logger.debug(f"{value_pred.shape}, {reward.shape}, {last_value_pred.shape}")

            # reward = reward.reshape(Args.TRAIN_ARGS['batch_size'], 1)

            # logger.debug(f"{value_pred.shape}, {reward.shape}, {last_value_pred.shape}")

            target = torch.add(reward, Args.TRAIN_ARGS['q_learning_discount'] * next_value_pred)

            loss: torch.Tensor = self.loss_fn(value_pred, target)
            
            # record all loses for tracking experiment
            self.total_loses += loss.item()

            loss.backward()
            self.optimizer.step()

            # for name, parameter in nnet.named_parameters():
            #     if parameter.requires_grad:
            #         print(f"Parameter: {name}, Value snippet: {parameter.data.cpu().numpy().flatten()[:1]}")


    
    def predict(self, state_queue: torch.Tensor) -> torch.Tensor:
        return self._nnet(state_queue)

    def get_loss(self):
        return self.total_loses

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
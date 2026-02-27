# log settings
from log_setting import MyLogging

logger = MyLogging.get_root_logger()

import torch
from torch import nn
from torch.nn import functional

class Q_network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ELU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32*240*256, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=12),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.normalize(x)
        x = self.conv_block(x)
        # logger.debug(x.shape)
        x = x.flatten(start_dim=2)
        # logger.debug(x.shape)
        x = x.permute(dims=(0, 2, 1))
        # logger.debug(x.shape)
        x = x.flatten(start_dim=1)
        # logger.debug(x.shape)
        x = self.classifier(x)
        # logger.debug(x)

        return x

if __name__ == "__main__":
    Q_model = Q_network()
    dummy_input = torch.randn((3,240,256), dtype=torch.float).unsqueeze(dim=0)
    logger.debug(dummy_input.shape)

    dummy_output = Q_model(dummy_input)
    logger.debug(dummy_output.shape)

    dummy_output = functional.softmax(dummy_output, dim=1)
    logger.debug(dummy_output.shape)

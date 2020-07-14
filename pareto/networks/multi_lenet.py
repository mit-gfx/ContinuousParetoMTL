from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiLeNet(nn.Module):
    def __init__(self) -> None:
        super(MultiLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, (5, 5))
        self.conv2 = nn.Conv2d(10, 20, (5, 5))
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        self.fc3_1 = nn.Linear(50, 10)
        self.fc3_2 = nn.Linear(50, 10)

    def shared_parameters(self) -> List[Tensor]:
        return [p for n, p in self.named_parameters() if not n.startswith('fc3')]

    def forward(
            self,
            x: Tensor,
        ) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = (self.fc3_1(x), self.fc3_2(x))
        return x

from typing import Tuple
import numpy as np
from torch import nn
import torch

from icecream import ic

class CNN(nn.Module):
    
    def __init__(self, input_shape: Tuple[int], n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, input_x):
        return self.net(input_x)
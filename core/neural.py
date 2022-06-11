from typing import Tuple
import numpy as np
from torch import nn
import torch

class CNN(nn.Module):
    
    def __init__(self, input_shape: Tuple[int], n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))


    def forward(self, input_x):
        conv_out = self.conv(input_x.squeeze(-1).float())
        return self.head(conv_out)

class DuelingCNN(nn.Module):
    """CNN network with duel heads for val and advantage."""
    
    def __init__(self, input_shape: Tuple[int], n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        #advantage head
        self.head_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        #value head
        self.head_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, input_x):
        adv, val = self.adv_val(input_x)
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val
    
    def adv_val(self, input_x):
        """Gets the advantage and value by passing out of the base network through the value and advantage heads.
        Args:
            input_x: input to network
        Returns:
            advantage, value
        """
        conv_out = self.conv(input_x.squeeze(-1).float())
        return self.head_adv(conv_out), self.head_val(conv_out)
from typing import Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layer import NoisyLinear

from icecream import ic

class CNN(nn.Module):
    """A simple CNN network.
    Args:
        input_shape: the number of inputs to the neural network
        n_actions: the number of outputs of the neural network
    """
    
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
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            NoisyLinear(1024, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions),
        )
        
        #value head
        self.head_val = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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

class NoisyDuelingCNN(nn.Module):
    """Noisy CNN network with duel heads for val and advantage."""
    
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
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            NoisyLinear(1024, 512),
            nn.ReLU(),
            NoisyLinear(512, n_actions),
        )
        
        #value head
        self.head_val = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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

class DistD3QN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
    ):
        """Initialization."""
        super(DistD3QN, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim[0], out_channels=256, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(in_dim)
        
        # set advantage layer
        self.advantage_hidden_layer = nn.Linear(conv_out_size, 2048)
        self.advantage_layer = nn.Linear(2048, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = nn.Linear(conv_out_size, 2048)
        self.value_layer = nn.Linear(2048, atom_size)
    
    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        conv_out = self.conv(x.squeeze(-1))
        adv_hid = F.relu(self.advantage_hidden_layer(conv_out))
        val_hid = F.relu(self.value_hidden_layer(conv_out))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

class RainbowDQN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
        sigma: float = 0.5,
    ):
        """Initialization."""
        super(RainbowDQN, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        conv_out_size = self._get_conv_out(in_dim)
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(conv_out_size, 512, sigma)
        self.advantage_layer = NoisyLinear(512, out_dim * atom_size, sigma)

        # set value layer
        self.value_hidden_layer = NoisyLinear(conv_out_size, 512, sigma)
        self.value_layer = NoisyLinear(512, atom_size, sigma)
    
    def _get_conv_out(self, shape) -> int:
        """Calculates the output size of the last conv layer.
        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        conv_out = self.conv(x.squeeze(-1))
        adv_hid = F.relu(self.advantage_hidden_layer(conv_out))
        val_hid = F.relu(self.value_hidden_layer(conv_out))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

# nn.Conv2d(in_channels=in_dim[0], out_channels=64, kernel_size=9, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.AvgPool2d(3, 1),
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.AvgPool2d(3, 1),
#             nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),

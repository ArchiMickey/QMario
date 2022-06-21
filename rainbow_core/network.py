import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .layer import NoisyLinear

from icecream import ic

class RainbowDQN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
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
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(3136, 512)
        self.advantage_layer = NoisyLinear(512, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(3136, 512)
        self.value_layer = NoisyLinear(512, atom_size)
    
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
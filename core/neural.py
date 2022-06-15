from typing import Tuple
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
import torch
import math

class NoisyLinear(nn.Linear):
    """Noisy Layer using Independent Gaussian Noise.
    based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/
    Chapter08/lib/dqn_extra.py#L19
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True):
        """
        Args:
            in_features: number of inputs
            out_features: number of outputs
            sigma_init: initial fill value of noisy weights
            bias: flag to include bias to linear layer
        """
        super().__init__(in_features, out_features, bias=bias)

        weights = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(weights)
        epsilon_weight = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", epsilon_weight)

        if bias:
            bias = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(bias)
            epsilon_bias = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", epsilon_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """initializes or resets the paramseter of the layer."""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input_x: Tensor) -> Tensor:
        """Forward pass of the layer.
        Args:
            input_x: input tensor
        Returns:
            output of the layer
        """
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        noisy_weights = self.sigma_weight * self.epsilon_weight.data + self.weight

        return F.linear(input_x, noisy_weights, bias)



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
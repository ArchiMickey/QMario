from typing import List
import numpy as np
import torch
from torch import nn, Tensor
from abc import ABC
from icecream import ic

class Agent(ABC):
    """Basic agent that always returns 0."""

    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, state: Tensor, device: str, *args, **kwargs) -> List[int]:
        """Using the given network, decide what action to carry.
        Args:
            state: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        return [0]

class ValueAgent(Agent):
    def __init__(
        self,
        net: nn.Module,
        action_space: int,
        eps_start: float = 1.0,
        eps_end: float = 0.2,
        eps_frames: float = 1000,
    ):
        super().__init__(net)
        self.action_space = action_space
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

    @torch.no_grad()
    def __call__(self, state: Tensor, device: str) -> List[int]:
        """Takes in the current state and returns the action based on the agents policy.
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by policy
        """
        if not isinstance(state, list):
            state = [state]

        if np.random.random() < self.epsilon:
            action = self.get_random_action(state)
        else:
            action = self.get_action(state, device)

        return action

    def get_random_action(self, state: Tensor) -> int:
        """returns a random action."""
        actions = []

        for i in range(len(state)):
            action = np.random.randint(0, self.action_space)
            actions.append(action)

        return actions

    def get_action(self, state: Tensor, device: torch.device):
        """Returns the best action based on the Q values of the network.
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            action defined by Q values
        """
        if not isinstance(state, Tensor):
            state = np.array(state)
            state = torch.tensor(state, device=device)

        state = state.unsqueeze(0).squeeze(-1)
        ic(state.shape)
        
        q_values = self.net(state)
        # ic()
        # ic(q_values)
        _, actions = torch.max(q_values, dim=1)
        return actions.detach().cpu().numpy()

    def update_epsilon(self, step: int) -> None:
        """Updates the epsilon value based on the current step.
        Args:
            step: current global step
        """
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)

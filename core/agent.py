from typing import Tuple
import gym, torch
import numpy as np
from torch import nn
import wandb
from .memory import ReplayBuffer, Experience

class Mario:
    def __init__(self, env: gym.Env, replay_buffer:ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()
    
    def reset(self):
        self.state = self.env.reset()
    
    def get_action(self, net:nn.Module, epsilon: float, device: str):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])
            if device not in ['cpu']:
                state = state.cuda(device)
            
            q_value = net(state)
            _, action = torch.max(q_value, dim=1)
            action = int(action.item())
        
        return action
    
    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        action = self.get_action(net, epsilon, device)
        new_state, reward, done, _ = self.env.step(action)
        
        self.env.render()
        
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        
        return reward, done
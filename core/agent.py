from typing import Tuple
import gym, torch
import numpy as np
from torch import nn
from .replay_buffer import ReplayBuffer, Experience
from icecream import ic
class Agent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
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
            state = np.array(self.state)
            state = torch.Tensor(state)
            
            if device not in ['cpu']:
                state = state.cuda(device)
                
            state = state.unsqueeze(0)
            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
            
            # ic()
            # ic(action)
        
        return action
    
    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu") -> Tuple[float, bool]:
        action = self.get_action(net, epsilon, device)
        new_state, reward, done, _ = self.env.step(action)
        
        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        
        return reward, done
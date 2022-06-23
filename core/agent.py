from collections import deque
from typing import Tuple
import gym, torch
import numpy as np
from torch import nn
from .replay import PERBuffer, Experience, MultiStepBuffer
from icecream import ic
class Agent:
    def __init__(self, env: gym.Env, buffer: PERBuffer, use_n_step: bool = False, buffer_n: MultiStepBuffer = None) -> None:
        self.env = env
        self.buffer = buffer
        self.use_n_step = use_n_step
        if self.use_n_step:
            self.buffer_n = buffer_n
        self.reset()
        self.state = self.env.reset()
    
    def reset(self):
        self.state = self.env.reset()
    
    def get_action(self, net:nn.Module, epsilon: float, device: str):
        if np.random.random() < 0:
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
        if self.use_n_step:
            # TODO implement n-step
            wait_list = deque(maxlen=3)
            wait_list.append(exp)
            self.buffer_n.append(exp)
            if len(self.buffer_n) > len(self.buffer):
                self.buffer.append(exp)
            
        else:
            self.buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        
        return reward, done
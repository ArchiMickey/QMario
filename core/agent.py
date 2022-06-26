from collections import deque
from typing import Tuple
import gym, torch
import numpy as np
from torch import nn
from .replay import PERBuffer, Experience, MultiStepBuffer

from icecream import ic
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT

class Agent:
    def __init__(self, env: gym.Env, buffer: PERBuffer, use_n_step: bool = False, buffer_n: MultiStepBuffer = None) -> None:
        self.env = env
        self.buffer = buffer
        self.use_n_step = use_n_step
        if self.use_n_step:
            self.buffer_n = buffer_n
        self.reset()
        self.state = self.env.reset()
        self.wait_list = deque(maxlen=10)
            
    
    def reset(self):
        self.state = self.env.reset()
    
    def get_action(self, net:nn.Module, epsilon: float, device: str):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = np.array(self.state) # state.shape = [4, 84, 84, 1]
            state = torch.FloatTensor(state)
            
            if device not in ['cpu']:
                state = state.cuda(device)
                
            q_values = net(state.unsqueeze(0))
            action = q_values.argmax()
            action = action.detach().cpu().numpy()
            action = int(action.item())
        
        return action, q_values
    
    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu") -> Tuple[float, bool]:
        action, q_values = self.get_action(net, epsilon, device)
        new_state, reward, done, _ = self.env.step(action)
        exp = Experience(self.state, action, reward, done, new_state)
        if self.use_n_step:
            self.wait_list.append(exp)
            self.buffer_n.append(exp)
            while len(self.buffer) < len(self.buffer_n):
                self.buffer.append(self.wait_list.popleft())
            
        else:
            self.buffer.append(exp)
        self.state = new_state
        if done:
            self.reset()
        
        return reward, done
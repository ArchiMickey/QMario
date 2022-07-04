from collections import deque
from typing import Tuple
import gym, torch
import numpy as np
from torch import nn
from .replay import PERBuffer, Experience, MultiStepBuffer

from icecream import ic
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT

class Agent:
    def __init__(self, env: gym.Env, buffer: PERBuffer, use_n_step: bool = False, buffer_n: MultiStepBuffer = None, episode_length: int = 200) -> None:
        self.env = env
        self.buffer = buffer
        self.use_n_step = use_n_step
        self.episode_length = episode_length
        if self.use_n_step:
            self.buffer_n = buffer_n
        self.reset()
        self.state = self.env.reset()
        self.wait_list = deque(maxlen=10)
        self.curr_score = 0
        self.curr_episode_step = 0
            
    
    def reset(self):
        self.state = self.env.reset()
        self.curr_score = 0
        self.curr_episode_step = 0
    
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
        
        return action
    
    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu") -> Tuple[float, bool]:
        action = self.get_action(net, epsilon, device)
        new_state, reward, done, info = self.env.step(action)
        self.curr_episode_step += 1
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        
        episode_end = done or self.curr_episode_step % self.episode_length == 0
        
        if episode_end:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        
        reward /= 10.
        
        exp = Experience(self.state, action, reward, done, new_state)
        if self.use_n_step:
            self.wait_list.append(exp)
            self.buffer_n.append(exp)
            while len(self.buffer) < len(self.buffer_n):
                self.buffer.append(self.wait_list.popleft())
        else:
            self.buffer.append(exp)
            
        self.state = new_state
        
        if episode_end:
            self.reset()
        
        return reward, done
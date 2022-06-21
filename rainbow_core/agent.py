import gym
import numpy as np
import torch
from .replay import ReplayBuffer, PrioritizedReplayBuffer
from torch import nn

from icecream import ic

class Agent:
    def __init__(self, env: gym.Env, memory: PrioritizedReplayBuffer, memory_n: ReplayBuffer, transisiton: list,
                 is_test: bool = False, use_n_step: bool = False) -> None:
        self.env = env
        self.memory = memory
        self.memory_n = memory_n
        self.transisiton = transisiton
        self.is_test = is_test
        self.use_n_step = use_n_step
        self.state = self.env.reset()
    
    def reset(self):
        self.state = self.env.reset()
    
    def select_action(self, net: nn.Module, device, deterministic=False) -> np.ndarray:
        state = torch.FloatTensor(self.state)
        state = state.unsqueeze(0)
        selected_action = net(state.to(device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not deterministic:
            self.transisiton = [state, selected_action]
        
        return selected_action.item()
    
    def step(self, action: np.ndarray):
        next_state, reward, done, _ = self.env.step(action)
        if not self.is_test:
            self.transisiton += [reward, next_state, done]
            
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transisiton)
            
            else:
                one_step_transition = self.transisiton
                        
            if one_step_transition:
                
                self.memory.store(*one_step_transition)
        self.state = next_state
        
        if done:
            self.reset()
                
        return reward, done
        
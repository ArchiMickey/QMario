
from collections import deque
import torch
from core.utils import make_mario
from .env_wrapper import EnvPreprocess

class MarioConfig():
    def __init__(self):        
        self.state_dim = (4, 84, 84)
        self.action_dim = None
        self.save_dir = None
        
        self.cuda = None
                
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.save_iter = 1e5
        
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.9
        
        self.train_iter = 500000
        self.burnin = 1e4
        self.learn_iter = 3
        self.sync_iter = 1e4
    
    def new_game(self):
        env = make_mario()
        return EnvPreprocess(env)

game_config = MarioConfig()
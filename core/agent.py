from collections import deque
from pyexpat import model
from matplotlib.pyplot import axis
import torch
import numpy as np
import random

from .DDQN import MarioNet

class Mario:
    def __init__(self, config):
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.save_dir = config.save_dir
        
        # Use cuda when available
        self.cuda = torch.cuda.is_available()
        
        # DNN for the learning process
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.cuda:
            self.net = self.net.to(device = 'cuda')
        
        # For every state, the agent can choose an action by its experience or random decision
        # exploration rate is the chance of using random actions
        # if it exploits, it chooses the actions by inputting the state into the DNN
        self.exploration_rate = config.exploration_rate
        self.exploration_rate_decay = config.exploration_rate_decay
        self.exploration_rate_min = config.exploration_rate_min
        
        self.curr_step = 0
        
        self.save_iter = config.save_iter
        
        self.memory = config.memory
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = config.burnin
        self.learn_iter = config.learn_iter
        self.sync_iter = config.sync_iter
        
    def td_estimate(self, state, action):
        curr_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ] # Q_online(s, a)
        return curr_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis = 1)
        next_Q = self.net(next_state, model = "target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_iter)}.ckpt"
        )
        torch.save(
            dict(model = self.net.state_dict(), exploration_rate = self.exploration_rate),
                 save_path,
        )
        print(f"MarioNet is saved to {save_path} at step {self.curr_step}")
    
    def action(self, state):
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        
        # EXPLOIT
        else:
            state = state.__array__()
            if self.cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values,axis = 1).item()
        
        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # add the step number and return the decision
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        # store the experience to self.memory(??)
        state = state.__array__()
        next_state = next_state.__array__()
        
        if self.cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])
        
        self.memory.append((state, next_state, action, reward, done, ))
            
    def recall(self):
        # retrieve a batch of experiences from memory(??)
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def train(self):
        if self.curr_step % self.sync_iter == 0:
            self.sync_Q_target()
        
        if self.curr_step % self.save_iter == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_iter != 0:
            return None, None
        
        state, next_state, action, reward, done = self.recall()
        
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        
        loss = self.update_Q_online(td_est, td_tgt)
        
        return (td_est.mean().item(), loss)
    
if __name__ == "__main__":
    net = MarioNet
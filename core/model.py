from collections import OrderedDict
import copy
from turtle import forward
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor, nn, true_divide
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import wandb
from .memory import ReplayBuffer, RLDataset
from .agent import Mario
from .utils import make_mario

class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size:int = 128):
        c, h, w = obs_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:   
            raise ValueError(f"Expecting input width: 84, got: {w}")
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    def forward(self, state):
        return self.net(state.float())

class DDQN(nn.Module):
    def __init__(self, obs_dim, n_actions) -> None:
        super().__init__()
        c, h, w = obs_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
class DQNLightning(LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.00025,
        env: str = 'SuperMarioBros-v0',
        gamma: float = 0.9,
        sync_rate: int = 10000,
        replay_size: int = 100000,
        warm_start_size: int = 1000,
        eps_decay: int = 0.99999975,
        eps_start: float = 1.0,
        eps_min: float = 0.1,
        episode_length: int = 1000,
        warm_start_steps: int = 100000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.env = make_mario(env)
        self.env.reset()
        obs_dim = self.env.observation().shape
        n_actions = self.env.action_space.n
        
        self.net = DQN(obs_dim, n_actions)
        self.target_net = DQN(obs_dim, n_actions)
        
        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Mario(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)
    
    def populate(self, steps: int = 1000):
        for i in range(steps):
            print(f"warming up at step {i+1}", end='\r')
            self.agent.play_step(self.net, epsilon=1)
    
    def forward(self, state: Tensor):
        return self.net(state).float()
    
    def loss_fn(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        # ?????????????????????
        states, actions, rewards, dones, next_states = batch
        
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        
        return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        device = self.get_device(batch)
        self.hparams.eps_start *= self.hparams.eps_decay
        epsilon = max(self.hparams.eps_min, self.hparams.eps_start)
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        loss = self.loss_fn(batch)
        
        if self.trainer.strategy in {"ddp", "dp"}:
            loss = loss.unsqueeze(0)
        
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        self.log("my loss", loss, on_epoch=True)
        self.log("reward", reward, on_epoch=True)
        self.log("epsilon", epsilon, on_epoch=True)
        
        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
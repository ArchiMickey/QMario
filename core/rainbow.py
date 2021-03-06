from collections import deque
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, OrderedDict, List, Tuple
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .replay import MultiStepBuffer, PrioritisedReplayBuffer, Rainbow_RLDataset
from .mario_env import make_mario
from .network import RainbowDQN
from .agent import Agent
from .log import log_video

from torch import nn
from moviepy.editor import *

from icecream import ic

class RainbowLightning(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.00025,
        min_lr: float = 1e-8,
        env: str = 'SuperMarioBros-v0',
        gamma: float = 0.99,
        target_update: int = 10000,
        memory_size: int = 10000,
        warm_start_size: int = 1000,
        episode_length: int = 200,
        eps_start: float = 1,
        eps_decay: float = 0.9999,
        eps_end = 0.02,
        # NoisyNet parameters
        sigma: float = 0.5,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = -50,
        v_max: float = 50,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 1,
        # Video logging
        save_video: bool = False,
        fps: int = 20,
        # Logging
        avg_cut_off: int = 100,
        test_episode_interval: int = 20,
    ) -> None:
        super().__init__()
        
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.env_name = env
        self.gamma = gamma
        self.target_update = target_update
        self.memory_size = memory_size
        self.warm_start_size = warm_start_size
        self.episode_length = episode_length
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.n_step = n_step
        self.save_video = save_video
        self.fps = fps
        self.avg_cut_off = avg_cut_off
        self.test_episode_interval = test_episode_interval
        
        self.save_hyperparameters()
                
        self.env = make_mario(env_name=env)
        self.test_env = make_mario(env_name=env)
        self.obs_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        
        self.buffer = PrioritisedReplayBuffer(memory_size, alpha=alpha, beta_start=beta)
        
        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step:
            self.buffer_n = MultiStepBuffer(memory_size, n_steps=n_step, gamma=gamma)
        else:
            self.memory_n = None
                
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).cuda()
        
        self.net = RainbowDQN(self.obs_dim, self.action_dim, self.atom_size, self.support, self.sigma)
        self.target_net = RainbowDQN(self.obs_dim, self.action_dim, self.atom_size, self.support, self.sigma)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        self.use_noisy = False
        
        self.agent = Agent(self.env, self.buffer, self.use_n_step, self.buffer_n, self.episode_length)
        
        self.test_rewards = deque(maxlen=self.avg_cut_off)
        self.episode_rewards = deque(maxlen=self.avg_cut_off)
        for _ in range(self.avg_cut_off):
            self.test_rewards.append(0)
            self.episode_rewards.append(0)
        
        self.curr_reward: int = 0
        self.curr_episode_step: int = 0
        self.total_steps: int = 0
        self.total_episodes: int = 0
        
        self.last_episode_reward: int = 0
            
    def populate(self):
        """Carries out some episodes of the environment to fill the replay buffer before training.
        Args:
            steps: the number of steps for populating.
        """
        i = 0
        while(len(self.buffer) < self.batch_size):
            self.agent.play_step(self.net, 0, self.device)
        self.agent.reset()
    
    def run_1_episode(self) -> float:
        """Carries out N episodes of the environment with the current agent.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        frames = []
        durations = []
            
        self.agent.reset()
        episode_end = False
        episode_reward = 0
        episode_end = False
        step = 0
        
        frames.clear()
        durations.clear()
        
        while not episode_end:
            reward, done = self.agent.play_step(self.net, 0, self.device)
            step += 1
            episode_end = done or step % self.episode_length == 0
            episode_reward += reward
            frame = self.env.render(mode='rgb_array')
            frame = np.array(frame)
            if self.save_video:
                frames.append(frame)
                durations.append(1/self.fps)

        if self.save_video:
            log_video(self.env_name, frames, durations, self.total_steps, episode_reward, self.fps)
        
        self.test_env.reset()
        frames.clear()
        durations.clear()

        return episode_reward
        
    def forward(self, state: Tensor):
        return self.net(state).float()
    
    def _compute_dqn_loss(self, exps, gamma: float):
        state, action, reward, done, next_state = exps
        state = state.float()
        done = done.unsqueeze(-1).float()
        reward = reward.unsqueeze(-1).float()
        next_state = next_state.float()
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.net(next_state).argmax(1)
            next_dist = self.target_net.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size, device=self.device
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.net.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action.long()])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
    
    def loss_fn(self, exps, weights, n_exps):
        elementwise_loss = self._compute_dqn_loss(exps, self.gamma)
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            elementwise_loss_n_loss = self._compute_dqn_loss(n_exps, gamma)
            loss = torch.mean(elementwise_loss_n_loss * weights, dtype=torch.float32)
            return loss, elementwise_loss_n_loss
        loss = torch.mean(elementwise_loss * weights, dtype=torch.float32)
        return loss, elementwise_loss
    
    def on_train_start(self) -> None:
        self.populate()
        return super().on_train_start()
    
    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        reward, done = self.agent.play_step(self.net, 0, self.device)
        self.total_steps += 1
        self.curr_episode_step += 1
        self.log("total_steps", self.total_steps, prog_bar=True)
        self.curr_reward += reward
        self.log("curr_reward", self.curr_reward, prog_bar=True)
        if self.use_n_step:
            exps, indices, weights, n_exps = batch
        else:
            exps, indices, weights  = batch
            n_exps = None
        
        loss, elementwise_loss = self.loss_fn(exps, weights, n_exps)
        if self.trainer.strategy in {"ddp", "dp"}:
            loss = loss.unsqueeze(0)
        
        # Soft update of target network
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        episode_end = done or self.curr_episode_step % self.episode_length == 0
        
        if episode_end:
            self.total_episodes += 1
            self.episode_rewards.append(self.curr_reward)
            self.log("episode_reward", self.curr_reward)
            self.log("avg_episode_reward", np.mean(self.episode_rewards))
            self.curr_reward = 0
            self.curr_episode_step = 0
            
            if self.total_episodes % self.test_episode_interval == 0:
                print("Testing...")
                self.run_1_episode()
                test_reward = self.run_1_episode()
                self.test_rewards.append(test_reward)
                self.log("test_reward", test_reward)
                self.log("avg_test_reward", np.mean(self.test_rewards))
            
        self.log("curr_reward", self.curr_reward, prog_bar=True)
        self.log("total_episodes", self.total_episodes, prog_bar=True)
        self.log_dict(
            {
                "loss": loss,
                "step_reward": reward,
            },
        )
        
        self.lr_schedulers().step()
        
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        indices = indices.int().detach().cpu().numpy()
        self.buffer.update_priorities(indices, new_priorities)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        self.net.reset_noise()
        self.target_net.reset_noise()
    
    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_1_episode(self.test_env, 10, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("self.last_test_reward", avg_reward)
        return {"self.last_test_reward": avg_reward}
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr, eps=1.5e-4, capturable=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                         T_0=self.target_update,
                                                         eta_min=self.min_lr
                                                         )
            }
        }

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = Rainbow_RLDataset(self.buffer, self.batch_size, self.buffer_n)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=6,
            drop_last=True
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()
    
    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0][0].device.index if self.on_gpu else "cpu"

if __name__ == "__main__":
    from pytorch_lightning import Trainer
    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=10,
    )
    model = RainbowLightning(
        batch_size=16,
        episode_length=20,
    )
    ic(model)
    trainer.fit(model=model)
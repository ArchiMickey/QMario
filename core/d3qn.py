from collections import deque
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, OrderedDict, List, Tuple
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .replay_buffer import MultiStepPERBuffer, PER_RLDataset
from .agent import Agent
from .env_wrapper import make_mario
from .loss import per_ddqn_loss

from torch import nn
import cv2
from moviepy.editor import *
import wandb

from .neural import DuelingCNN

from icecream import ic

class D3QNLightning(pl.LightningModule):
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.00025,
        env: str = 'SuperMarioBros-v0',
        gamma: float = 0.9,
        sync_rate: int = 10000,
        replay_size: int = 30000,
        warm_start_size: int = 1000,
        eps_decay: int = 0.999,
        eps_start: float = 1,
        eps_min: float = 0.1,
        n_steps: int = 1,
        episode_length: int = 1024,
        avg_rewards_len: int = 100,
        save_video: bool = False,
        fps: int = None,
        video_rate: int = 50,
    ) -> None:
        super().__init__()
        
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.eps_decay = eps_decay
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.n_steps = n_steps
        self.episode_length = episode_length
        self.save_video = save_video
        self.fps = fps
        self.video_rate = video_rate
        
        self.save_hyperparameters()
        
        self.env = make_mario(env)
        self.test_env = make_mario(env)
        self.env.reset()
        self.test_env.reset()
        
        obs_dim = self.env.observation_space.shape
        n_actions = self.env.action_space.n
        
        self.net = DuelingCNN(obs_dim, n_actions)
        self.target_net = DuelingCNN(obs_dim, n_actions)
        
        for p in self.target_net.parameters():
            p.requires_grad = False
        
        self.buffer = MultiStepPERBuffer(self.replay_size, n_steps=n_steps)
        self.agent = Agent(self.env, self.buffer)
        self.test_agent = Agent(self.test_env, self.buffer)
        
        self.total_rewards = deque(maxlen=avg_rewards_len)
        self.episode_reward: float = 0
        self.total_episodes: int = 0
        
        self.avg_reward_len = avg_rewards_len
        for _ in range(avg_rewards_len):
            self.total_rewards.append(torch.tensor(0, device=self.device))
        self.avg_rewards = float(np.mean(list(self.total_rewards)))
        
        self.populate(self.hparams.warm_start_size)
        
        self.recording = True
        self.frame_ls = []
        self.duration_ls = [] # For recording videos
    
    def populate(self, steps: int = 1000):
        for i in range(steps):
            print(f"warming up at step {i+1}", end='\r')
            self.agent.play_step(self.net, epsilon=1)
        self.env.reset()
    
    def run_n_episodes(self, env, n_episodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """Carries out N episodes of the environment with the current agent.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []

        for episode in range(n_episodes):
            self.test_env.reset()
            done = False
            episode_reward = 0
            
            self.frame_ls.clear()
            self.duration_ls.clear()
            
            while not done:
                reward, done= self.test_agent.play_step(self.net, 0, self.device)
                episode_reward += reward
                frame = env.render('rgb_array')
                frame = np.array(frame)
                if self.save_video:
                    self.frame_ls.append(frame)
                    self.duration_ls.append(1/self.fps)
                # frame = cv2.resize(frame, (512, 480))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                cv2.imshow("QMario", frame)
                cv2.waitKey(20)

            if self.save_video:
                clip = ImageSequenceClip(self.frame_ls, durations=self.duration_ls)
                clip.write_videofile(f"test_video/mario_episode{episode}_reward{episode_reward}.mp4",
                                     fps=60,
                                     codec='mpeg4'
                                     )    

            total_rewards.append(episode_reward)

        return total_rewards
    
    def forward(self, state: Tensor):
        return self.net(state).float()
    
    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        
        device = self.get_device(batch)
        epsilon: float = max(self.hparams.eps_min, self.hparams.eps_start)
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.hparams.eps_start *= self.hparams.eps_decay
        self.episode_reward += reward
        
        samples, indices, weights = batch
        indices = indices.cpu().numpy()
        
        loss, batch_weights = per_ddqn_loss(samples, weights, self.net, self.target_net, self.gamma)
        
        if self.trainer.strategy in {"ddp", "dp"}:
            loss = loss.unsqueeze(0)
        
        self.buffer.update_priorities(indices, batch_weights)
        
        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        if self.recording:
            frame = self.env.render('rgb_array')
            frame = np.array(frame)
            self.frame_ls.append(frame)
            self.duration_ls.append(1/self.fps)
        
        if done:
            self.total_rewards.append(self.episode_reward)
            self.avg_rewards = float(np.mean(list(self.total_rewards)))
            self.log_dict(
            {
                "episode_reward": self.episode_reward,
                "total_episodes": self.total_episodes,
                "avg_reward": self.avg_rewards,
            }
            )
            
            if self.recording:
                clip = ImageSequenceClip(self.frame_ls, durations=self.duration_ls)
                clip.write_videofile(f"train_video/mario_episode{self.total_episodes}_reward{self.episode_reward}.mp4",
                                     fps=60,
                                     )
                wandb.log({f"episode{self.total_episodes}_gameplay": wandb.Video(f"train_video/mario_episode{self.total_episodes}_reward{self.episode_reward}.mp4")})
                self.frame_ls.clear()
                self.duration_ls.clear()
            
            self.total_episodes += 1
            if self.total_episodes % self.video_rate == 0:
                self.recording = True
            else:
                self.recording = False
            self.episode_reward = 0
            
        log = {
            "total_reward": torch.tensor(self.episode_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }

        self.log_dict(
            {
                "loss": loss,
                "step_reward": reward,
                "epsilon": epsilon,
            }
        )
        
        self.lr_schedulers().step()
        
        return OrderedDict({"loss": loss, "log": log})
    
    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 10, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}
    
    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            sum([p["params"] for p in optimizer.param_groups], []), gradient_clip_val
        )
        self.log("grad_norm", grad_norm)
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                         T_0=400000,
                                                         T_mult=2,
                                                         eta_min=1e-6
                                                         )
            }
        }

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = PER_RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=6,
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
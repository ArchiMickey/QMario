import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, OrderedDict, List, Tuple
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .replay import ReplayBuffer, PrioritizedReplayBuffer, RLDataset
from .mario_env import make_mario
from .network import RainbowDQN
from .agent import Agent

from torch import nn
import cv2
from moviepy.editor import *
import wandb

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
        episode_length: int = 1024,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 1,
        # Video loggint
        save_video: bool = False,
        fps: int = 20,
        video_rate: int = 50,
    ) -> None:
        super().__init__()
        
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.target_update = target_update
        self.memory_size = memory_size
        self.episode_length = episode_length
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.n_step = n_step
        self.save_video = save_video
        self.fps = fps
        self.video_rate = video_rate
        
        self.save_hyperparameters()
        
        self.env = make_mario(env_name=env)
        self.test_env = make_mario(env_name=env)
        obs_dim = self.env.observation_space.shape
        action_dim = self.env.action_space.n
        
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, episode_length, alpha=alpha, n_step=self.n_step)
        
        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, episode_length, n_step=n_step, gamma=gamma
            )
        else:
            self.memory_n = None
        
        self.transition = list()
        
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        )
        
        self.net = RainbowDQN(obs_dim, action_dim, self.atom_size, self.support)
        self.target_net = RainbowDQN(obs_dim, action_dim, self.atom_size, self.support)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()
        
        self.is_test = False
        self.agent = Agent(self.env, self.memory, self.memory_n, self.transition, self.is_test, self.use_n_step)
        self.test_agent = Agent(self.test_env, self.memory, None, self.transition, True)
        
        self.episode_reward: float = 0
        self.total_episodes: int = 0
        
        self.populate()
            
    def populate(self):
        """Carries out some episodes of the environment to fill the replay buffer before training.
        Args:
            steps: the number of steps for populating.
        """
        i = 0
        while(len(self.memory) < self.episode_length):
            print(f"warming up at step {i+1}", end='\r')
            action = self.agent.select_action(self.net, 'cpu')
            self.agent.step(action)
            i += 1
        self.net.support = self.net.support.cuda()
        self.support = self.support.cuda()
    
    def run_n_episodes(self, env, n_episodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """Carries out N episodes of the environment with the current agent.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []
        frames = []
        durations = []
        for episode in range(n_episodes):
            
            self.test_env.reset()
            done = False
            episode_reward = 0
            
            frames.clear()
            durations.clear()
            
            while not done:
                action = self.agent.select_action(self.net, self.device)
                reward, done = self.agent.step(action)
                episode_reward += reward
                frame = self.env.render('rgb_array')
                frame = np.array(frame)
                if self.save_video:
                    frames.append(frame)
                    durations.append(1/self.fps)
                # frame = cv2.resize(frame, (512, 480))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # cv2.imshow("QMario", frame)
                # cv2.waitKey(20)

            if self.save_video:
                clip = ImageSequenceClip(frames, durations=durations)
                clip.write_videofile(f"test_video/mario_episode{self.total_episodes}_reward{episode_reward}.mp4",
                                     fps=self.fps,
                                     )
                wandb.log({f"gameplay": wandb.Video(f"test_video/mario_episode{self.total_episodes}_reward{episode_reward}.mp4")})
            self.test_env.reset()
            total_rewards.append(episode_reward)

        return total_rewards
    
    def forward(self, state: Tensor):
        return self.net(state).float()
    
    def _compute_nstep_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).cuda()
        next_state = torch.FloatTensor(samples["next_obs"]).cuda()
        action = torch.LongTensor(samples["acts"]).cuda()
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).cuda()
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).cuda()
        exps = (state, action, reward, next_state, done)
        return self._compute_dqn_loss(exps, gamma=gamma)
        
    
    def _compute_dqn_loss(self, exps, gamma: float):
        state, action, reward, next_state, done = exps
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.net(next_state).argmax(1)
            next_dist = self.target_net.dist(next_state)
            ic(len(self.memory), next_state.shape, next_action.shape)
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
    
    def loss_fn(self, exps, weights, indices):
        elementwise_loss = self._compute_dqn_loss(exps, self.gamma)
        
        loss = torch.mean(elementwise_loss * weights)
        
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices.cpu())
            elementwise_loss_n_loss = self._compute_nstep_loss(samples, gamma)
            elementwise_loss = elementwise_loss + elementwise_loss_n_loss
            
            loss = torch.mean(elementwise_loss * weights)
        return loss, elementwise_loss
    
    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        device = self.get_device(batch)
        
        
        
        action = self.agent.select_action(self.net, device)
        reward, done = self.agent.step(action)
        
        self.episode_reward += reward
        
        exps, weights, indices = batch
        loss, elementwise_loss = self.loss_fn(exps, weights, indices)
        
        if self.trainer.strategy in {"ddp", "dp"}:
            loss = loss.unsqueeze(0)
        
        # Soft update of target network
        if self.global_step % self.target_update == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        
        if done:
            self.log_dict(
            {
                "episode_reward": self.episode_reward,
                "total_episodes": self.total_episodes,
            }
            )
            if self.total_episodes % self.video_rate == 0:
                test_rewards = self.run_n_episodes(self.test_env, 1, 0)
                avg_test_reward = sum(test_rewards) / len(test_rewards)
                self.log("avg_test_reward", avg_test_reward)
            self.total_episodes += 1
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
            }
        )
        
        self.lr_schedulers().step()
        
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        return OrderedDict({"loss": loss, "log": log})
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: int = 0) -> None:
        self.net.reset_noise()
        self.target_net.reset_noise()
    
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
        optimizer = Adam(self.net.parameters(), lr=self.lr)
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
        dataset = RLDataset(self.memory, self.episode_length, self.beta)
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

if __name__ == "__main__":
    from pytorch_lightning import Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        gradient_clip_val=10,
    )
    model = RainbowLightning()
    ic(model)
    trainer.fit(model=model)
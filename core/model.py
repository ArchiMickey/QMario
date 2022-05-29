import copy
from turtle import forward
from typing import Dict, List, Optional, OrderedDict, Tuple
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data.dataloader import DataLoader
from .agent import ValueAgent
from .neural import CNN
from .memory import Experience, MultiStepBuffer
from .dataset import ExperienceSourceDataset
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, TransformObservation, RecordVideo
from pl_bolts.models.rl.common.gym_wrappers import MaxAndSkipEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from icecream import ic

class DDQN(pl.LightningModule):
    def __init__(
        self,
        env: str = 'SuperMarioBros-v0',
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_last_frame: int = 150000,
        sync_rate: int = 1000,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        replay_size: int = 100000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        seed: int = 123,
        batches_per_epoch: int = 1000,
        n_steps: int = 1,
        **kwargs,
    ):
        super().__init__()
        
        self.exp = None
        self.env = self.make_environment(env)
        self.test_env = self.make_environment(env)

        self.obs_shape = self.env.observation_space.shape
        ic(self.obs_shape)
        self.n_actions = self.env.action_space.n

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.net = None
        self.target_net = None
        self.build_networks()
        
        ic(self.net)
        
        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=1,
            eps_end=0.01,
            eps_frames=eps_last_frame,
        )
        
        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.batches_per_epoch = batches_per_epoch
        self.n_steps = n_steps
        
        self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len
        
        for _ in range(avg_reward_len):
            self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()
        
    def run_n_episodes(self, env, n_epsiodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """Carries out N episodes of the environment with the current agent.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
                action = self.agent(episode_state, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards
    
    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action = self.agent(self.state, self.device) # get the actions Q-value from the agent
                
                # ic()
                # ic(action)
                
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=next_state)
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()
    
    def build_networks(self) -> None:
        """Initializes the DQN train and target networks."""
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)
    
    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)
    
    def train_batch(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0
        
        while True:
            self.total_steps += 1
            action = self.agent.get_action(self.state, self.device)
            
            next_state, reward, done, _ = self.env.step(action[0])
            
            episode_reward += reward
            episode_steps += 1
            
            exp = Experience(state=self.state, action=action, reward=reward, done=done, new_state=next_state)
            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            
            if done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0
            
            states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)
            
            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break
    
    def double_dqn_loss(
    batch: Tuple[Tensor, Tensor],
    net: nn.Module,
    target_net: nn.Module,
    gamma: float = 0.99,
) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer. This uses an improvement to the original
        DQN loss by using the double dqn. This is shown by using the actions of the train network to pick the value
        from the target network. This code is heavily commented in order to explain the process clearly.

        Args:
            batch: current mini batch of replay data
            net: main training network
            target_net: target network of the main training network
            gamma: discount factor

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch  # batch of experiences, batch_size = 16

        actions = actions.long().squeeze(-1)

        state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # dont want to mess with gradients when using the target network
        with torch.no_grad():
            next_outputs = net(next_states)  # [16, 2], [batch, action_space]

            next_state_acts = next_outputs.max(1)[1].unsqueeze(-1)  # take action at the index with the highest value
            next_tgt_out = target_net(next_states)

            # Take the value of the action chosen by the train network
            next_state_values = next_tgt_out.gather(1, next_state_acts).squeeze(-1)
            next_state_values[dones] = 0.0  # any steps flagged as done get a 0 value
            next_state_values = next_state_values.detach()  # remove values from the graph, no grads needed

        # calc expected discounted return of next_state_values
        expected_state_action_values = next_state_values * gamma + rewards

        # Standard MSE loss between the state action values of the current state and the
        # expected state action values of the next state
        return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = self.double_dqn_loss(batch, self.net, self.target_net, self.gamma)

        if self._use_dp_or_ddp2(self.trainer):
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "train_loss": loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
            }
        )
    
    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}
    
    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]
    
    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)
    
    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()
    
    @staticmethod
    def make_environment(env_name: str, seed: Optional[int] = None):
        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, RIGHT_ONLY)
        env = RecordVideo(env, "test_video", lambda x: x % 1 == 0)
        env = MaxAndSkipEnv(env, skip=4)
        env = GrayScaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=84)
        env = TransformObservation(env, f = lambda x: x / 255.)
        env.reset()
        env = FrameStack(env, num_stack=4)
        return env

def cli_main():
    model = DDQN()

if __name__ == '__main__':
    cli_main()
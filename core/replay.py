from collections import namedtuple
from collections import deque
import numpy as np
from torch.utils.data.dataset import IterableDataset
from typing import Iterator, List, Tuple

from icecream import ic

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "done", "new_state"],
                        )

class _ReplayBuffer:
    """This version of replay buffer is deprecated. The new version is TODO.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        # print(len(self.buffer), batch_size)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )

class _MultiStepBuffer(_ReplayBuffer):
    """This version of N Step replay buffer is deprecated. The new version is TODO.
    """
    def __init__(self, capacity: int, n_steps: int = 1, gamma: float = 0.99):
        super().__init__(capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.history = deque(maxlen=self.n_steps)
        self.exp_history_queue = deque()
    
    def append(self, exp: Experience) -> None:
        """Add experience to the buffer.
        Args:
            exp: tuple (state, action, reward, done, new_state)
        """
        self.update_history_queue(exp)  # add single step experience to history
        while self.exp_history_queue:  # go through all the n_steps that have been queued
            experiences = self.exp_history_queue.popleft()  # get the latest n_step experience from queue

            last_exp_state, tail_experiences = self.split_head_tail_exp(experiences)

            total_reward = self.discount_rewards(tail_experiences)

            n_step_exp = Experience(
                state=experiences[0].state,
                action=experiences[0].action,
                reward=total_reward,
                done=experiences[0].done,
                new_state=last_exp_state,
            )

            self.buffer.append(n_step_exp)  # add n_step experience to buffer

    def update_history_queue(self, exp) -> None:
        """Updates the experience history queue with the lastest experiences. In the event of an experience step is
        in the done state, the history will be incrementally appended to the queue, removing the tail of the
        history each time.
        Args:
            env_idx: index of the environment
            exp: the current experience
            history: history of experience steps for this environment
        """
        self.history.append(exp)

        # If there is a full history of step, append history to queue
        if len(self.history) == self.n_steps:
            self.exp_history_queue.append(list(self.history))

        if exp.done:
            if 0 < len(self.history) < self.n_steps:
                self.exp_history_queue.append(list(self.history))

            # generate tail of history, incrementally append history to queue
            while len(self.history) > 2:
                self.history.popleft()
                self.exp_history_queue.append(list(self.history))

            # when there are only 2 experiences left in the history,
            # append to the queue then update the env stats and reset the environment
            if len(self.history) > 1:
                self.history.popleft()
                self.exp_history_queue.append(list(self.history))

            # Clear that last tail in the history once all others have been added to the queue
            self.history.clear()

    def split_head_tail_exp(self, experiences: Tuple[Experience]) -> Tuple[List, Tuple[Experience]]:
        """Takes in a tuple of experiences and returns the last state and tail experiences based on if the last
        state is the end of an episode.
        Args:
            experiences: Tuple of N Experience
        Returns:
            last state (Array or None) and remaining Experience
        """
        last_exp_state = experiences[-1].new_state
        tail_experiences = experiences

        if experiences[-1].done and len(experiences) <= self.n_steps:
            tail_experiences = experiences

        return last_exp_state, tail_experiences

    def discount_rewards(self, experiences: Tuple[Experience]) -> float:
        """Calculates the discounted reward over N experiences.
        Args:
            experiences: Tuple of Experience
        Returns:
            total discounted reward
        """
        total_reward = 0.0
        for exp in reversed(experiences):
            total_reward = (self.gamma * total_reward) + exp.reward
        return total_reward
    
    def sample_from_idxs(self, idxs: np.array):
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in idxs))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )

class _PERBuffer(_ReplayBuffer):
    """This version of PER replay buffer is deprecated. The new version is TODO.
    """
    def __init__(self, buffer_size, prob_alpha=0.6, beta_start=0.4, beta_frames=50000):
        super().__init__(capacity=buffer_size)
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_frames = beta_frames
        self.prob_alpha = prob_alpha
        self.capacity = buffer_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def update_beta(self, step) -> float:
        """Update the beta value which accounts for the bias in the PER.
        Args:
            step: current global step
        Returns:
            beta value for this indexed experience
        """
        beta_val = self.beta_start + step * (1.0 - self.beta_start) / self.beta_frames
        self.beta = min(1.0, beta_val)

        return self.beta

    def append(self, exp) -> None:
        """Adds experiences from exp_source to the PER buffer.
        Args:
            exp: experience tuple being added to the buffer
        """
        # what is the max priority for new sample
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp

        # the priority for the latest sample is set to max priority so it will be resampled soon
        self.priorities[self.pos] = max_prio

        # update position, loop back if it reaches the end
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32) -> Tuple:
        """Takes a prioritized sample from the buffer.
        Args:
            batch_size: size of sample
        Returns:
            sample of experiences chosen with ranked probability
        """
        # get list of priority rankings
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]

        # probability to the power of alpha to weight how important that probability it, 0 = normal distirbution
        probs = prios**self.prob_alpha
        probs /= probs.sum()

        # choise sample of indices based on the priority prob distribution
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        # samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))

        samples = (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states),
        )
        total = len(self.buffer)

        # weight of each sample datum to compensate for the bias added in with prioritising samples
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # return the samples, the indices chosen and the weight of each datum in the sample
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: List, batch_priorities: List) -> None:
        """Update the priorities from the last batch, this should be called after the loss for this batch has been
        calculated.
        Args:
            batch_indices: index of each datum in the batch
            batch_priorities: priority of each datum in the batch
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

class PER_RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: _ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        samples, indices, weights = self.buffer.sample(self.sample_size)
        states, actions, rewards, dones, new_states = samples
        for i in range(len(dones)):
            yield (states[i], actions[i], rewards[i], dones[i], new_states[i],), indices[i], weights[i]     

class Rainbow_RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: PERBuffer, sample_size: int = 200, buffer_n: MultiStepBuffer = None) -> None:
        self.buffer = buffer
        self.buffer_n = buffer_n
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        samples, indices, weights = self.buffer.sample(self.sample_size)
        states, actions, rewards, dones, new_states = samples
        n_states, n_actions, n_rewards, n_dones, n_new_states = self.buffer_n.sample_from_idxs(indices)
        for i in range(len(dones)):
            yield (states[i], actions[i], rewards[i], dones[i], new_states[i],), indices[i], weights[i], (n_states[i], n_actions[i], n_rewards[i], n_dones[i], n_new_states[i])
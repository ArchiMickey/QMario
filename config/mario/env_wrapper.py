import gym
from gym import Env, Wrapper
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
from torchvision import transforms as T
from gym.spaces import Box
import numpy as np
import torch

class SkipFrame(gym.Wrapper):
    # This Wrapper edits the step function to skip frames
    def __init__(self, env: Env, skip):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        # For these frames, actually the action are likely the same
        # Therefore we keep tracking the state
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

def EnvPreprocess(env):
    env = SkipFrame(env, skip=4)
    # gray scaled and resize to reduce computation time
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    # Put frames into a stack for feeding the learning model
    env = FrameStack(env, num_stack=4)
    return env
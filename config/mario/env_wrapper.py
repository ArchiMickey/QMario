import cv2
import gym
from gym import Env
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation
from gym.spaces import Box
import numpy as np

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
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation

def EnvPreprocess(env):
    env = SkipFrame(env, skip=4)
    # gray scaled and resize to reduce computation time
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    # Put frames into a stack for feeding the learning model
    env = FrameStack(env, num_stack=4)
    return env
# import the game
import gym_super_mario_bros
# Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import simplified actions
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from config.mario.env_wrapper import SkipFrame, ResizeObservation
from gym.wrappers import GrayScaleObservation, TransformObservation, FrameStack

import cv2

def make_mario(env_name: str):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, [['right'],
                            ['right', 'A']]
                      )
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f = lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env

def cv2render(state): 
    # a function to print the frames inside the frame stack
    for i in range(4):
        state_img = cv2.resize(state[i], (1000, 1000))
        state_img = cv2.cvtColor(state_img, cv2.COLOR_BGR2RGB)
        cv2.imshow("state", state_img)
        cv2.waitKey(1)
    return

if __name__ == '__main__':
    env = make_mario()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state},\n {reward},\n {done},\n {info}")
    print(f"{env.action_space.n}")
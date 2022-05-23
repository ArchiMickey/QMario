# import the game
import gym_super_mario_bros
# Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import simplified actions
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import cv2

def make_mario():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
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
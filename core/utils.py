# import the game
import gym_super_mario_bros
# Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import simplified actions
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def make_mario():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env.reset()
    return env

if __name__ == '__main__':
    env = make_mario()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state},\n {reward},\n {done},\n {info}")
    print(f"{env.action_space.n}")
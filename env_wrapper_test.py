# import the game
import gym_super_mario_bros
# Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import simplified actions
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, TransformObservation, FrameStack, ResizeObservation, RecordVideo
from pl_bolts.models.rl.common.gym_wrappers import MaxAndSkipEnv


def make_mario(env_name: str):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = RecordVideo(env, "wrapper_test_video", lambda x: True)
    env = MaxAndSkipEnv(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f = lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env

env = make_mario('SuperMarioBros-v0')

for i in range(5):
    env.reset()
    done = False
    while True:
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break
# import the game
import gym_super_mario_bros
# Joypad wrapper
from nes_py.wrappers import JoypadSpace
# import simplified actions
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import GrayScaleObservation, TransformObservation, FrameStack, ResizeObservation
import numpy as np
from pl_bolts.models.rl.common.gym_wrappers import MaxAndSkipEnv

import cv2
from PIL import Image
from moviepy.editor import *

from icecream import ic


def make_mario(env_name: str):
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = MaxAndSkipEnv(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f = lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env

env = make_mario('SuperMarioBros-v0')
vid_fps = 26

for i in range(5):
    frame_ls = []
    duration_ls = []
    env.reset()
    done = False
    while True:
        obs, reward, done, info = env.step(1)
        frame = env.render('rgb_array')
        frame = np.array(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_ls.append(frame)
        duration_ls.append(1/vid_fps)
        frame = cv2.resize(frame, (512, 480))
        # ic(curr_state.shape)
        # ic(type(curr_state))
        ic(len(frame_ls))
        cv2.imshow("QMario", frame)
        if cv2.waitKey(20) == ord('q'):
            exit()
        if done:
            break
    clip = ImageSequenceClip(frame_ls, durations=duration_ls)
    clip.write_videofile(f"test_video/mario_episode{i}.mp4", fps=60,)
import logging
from typing import List
import numpy as np
import torch
import wandb
from moviepy.editor import *
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from icecream import ic

def log_video(env_name: str, frames: List, durations: List, curr_steps: int, episode_reward: int, fps: int):
    clip = ImageSequenceClip(frames, durations=durations)
    clip.write_videofile(f"train_video/{env_name}/mario_step{curr_steps}_reward{episode_reward}.mp4",
                            fps=fps,
                            )
    wandb.log({f"gameplay": wandb.Video(f"train_video/{env_name}/mario_step{curr_steps}_reward{episode_reward}.mp4",
                                        caption=f"reward: {episode_reward}")})

def make_table():
    columns = ["state", "reward", "action"]
    for action in SIMPLE_MOVEMENT:
        columns += [f"{action}"]
    table = wandb.Table(columns=columns)
    return table
   

def log_table(table: wandb.Table, state: np.array, action: int, reward, q_values: torch.Tensor):
    log_q = tuple(q_values.detach().cpu().numpy())
    table.add_data(wandb.Image(state[-1]), reward, f"SIMPLE_MOVEMENT[action]", *log_q)
    return

if __name__ == '__main__':
    make_table()
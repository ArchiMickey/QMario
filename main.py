from datetime import datetime
from pathlib import Path

import torch
from core.agent import Mario
from core.log import MetricLogger
from config.mario import game_config
from core.cv2render import cv2render

save_dir = Path("checkpoints") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
game_config.save_dir = save_dir

game_config.cuda = torch.cuda.is_available()

if __name__ == '__main__':
    env = game_config.new_game()
    game_config.action_dim = env.action_space.n
    mario = Mario(game_config)
    logger = MetricLogger(save_dir)
    
    for episode in range(game_config.train_iter):
        
        state = env.reset()
        
        while True:
            # cv2render(state.__array__())
            action = mario.action(state)
            
            next_state, reward, done, info = env.step(action)
            
            mario.cache(state, next_state, action, reward, done)
            
            q, loss = mario.train()
            
            logger.log_step(reward, loss, q)
            
            state = next_state
            if done or info["flag_get"]:
                break
        
        logger.log_episode()
        if episode % 200 == 0:
            logger.record(episode, epsilon=mario.exploration_rate, step=mario.curr_step)
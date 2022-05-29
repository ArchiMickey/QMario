import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from core.model import DDQN
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
import wandb

checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="total_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{episode:02d}",
    every_n_train_steps=10000,
)

model = DDQN(
    batch_size=64,
    replay_size=3000,
    warm_start_size=3000,
    learning_rate=0.00025,
    eps_last_frame=1000,
    save_video=True,
    video_episode=100,
    sync_rate=10000,
)

wandb.init()
wandb.watch(model)
now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb_logger = WandbLogger(name=f"qMario-test-{now_dt}")
trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    logger=wandb_logger,
    max_epochs=40000,
    # val_check_interval=50,
    auto_lr_find=True,
    callbacks=[checkpoint_callback],
    benchmark=False
)

trainer.fit(model)
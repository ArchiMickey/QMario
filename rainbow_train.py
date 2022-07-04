import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from core.rainbow import RainbowLightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from datetime import datetime

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="episode_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{episode_reward}",
    every_n_train_steps=1000,
)

# 1 training step = 2.5 global step
Model = RainbowLightning(
    batch_size=32,
    lr=6.25e-5,
    min_lr=6.25e-5,
    gamma=0.9,
    target_update=10000,
    memory_size=100000,
    sigma=0.5,
    alpha=0.5,
    beta=0.4,
    v_min=-10,
    v_max=10,
    atom_size=51,
    n_step=3,
    save_video=True,
    fps=24,
    video_rate=10000,
)



now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

wandb_logger = WandbLogger(name=f"qMario-rainbow-{now_dt}")
wandb_logger.watch(Model)

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs=600000000,
    logger=wandb_logger,
    log_every_n_steps=5,
    gradient_clip_val= 10.0,
    auto_lr_find=True,
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
)

# trainer.tune(Model)
trainer.fit(Model)
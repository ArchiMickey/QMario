from pyexpat import model
from cv2 import triangulatePoints
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from rainbow_core.rainbow import RainbowLightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime
from icecream import ic

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="test_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{epoch:02d}",
    every_n_train_steps=1000,
    save_last=True,
)

model = RainbowLightning(
    batch_size=128,
    lr=2.5e-4,
    min_lr=1e-8,
    target_update=10000,
    memory_size=50000,
    episode_length=500,
    v_min=-200,
    v_max=200,
    atom_size=51,
    n_step=4,
    save_video=True,
    fps=24,
    video_rate=1,
)

now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(name=f"qMario-test-{now_dt}")
wandb.watch(model)

wandb_logger = WandbLogger()

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs=100,
    logger=wandb_logger,
    gradient_clip_val= 5.0,
    # val_check_interval=50,
    auto_lr_find=True,
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
)

# trainer.tune(model)
trainer.fit(model)
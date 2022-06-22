import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from rainbow_core.rainbow import RainbowLightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="avg_test_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{epoch:02d}",
    every_n_train_steps=1000,
    save_last=True,
)

Model = RainbowLightning(
    batch_size=256,
    lr=2.5e-5,
    min_lr=1e-8,
    gamma=0.95,
    target_update=10000,
    memory_size=25000,
    episode_length=4500,
    v_min=-50,
    v_max=50,
    atom_size=51,
    n_step=4,
    save_video=True,
    fps=24,
    video_rate=20,
)

now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

wandb_logger = WandbLogger(name=f"qMario-rainbow-{now_dt}", log_model="all")
wandb_logger.watch(Model, log='all')

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs=4000000,
    logger=wandb_logger,
    gradient_clip_val= 10.0,
    # val_check_interval=50,
    auto_lr_find=True,
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
)

# trainer.tune(model)
trainer.fit(Model)
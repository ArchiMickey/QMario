import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from core.d3qn import D3QNLightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime
from icecream import ic

checkpoint_callback = ModelCheckpoint(
    save_top_k=3,
    monitor="episode_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{epoch:02d}",
    every_n_train_steps=10000,
)

# test_model = DDQNLightning(
#     batch_size=32,
#     warm_start_steps=100,
#     episode_length=100,
#     replay_size=100,
# )

model = D3QNLightning(
    batch_size=128,
    lr=0.00025,
    warm_start_size=10000,
    episode_length=4500,
    gamma=0.9,
    n_steps=4,
    replay_size=100000,
    eps_start=1,
    eps_decay=0.9999,
    eps_min=0.01,
    sync_rate=250000,
    save_video=True,
    fps=24,
    video_rate=50,
)
# ic(model)

now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(name=f"qMario-d3qn-{now_dt}")
wandb.watch(model)

wandb_logger = WandbLogger()

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs=400000,
    logger=wandb_logger,
    gradient_clip_val= 15.0,
    # val_check_interval=50,
    auto_lr_find=True,
    auto_scale_batch_size="power",
    benchmark=False,
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
)

trainer.fit(model)
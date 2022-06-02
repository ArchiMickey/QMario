import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from core.model import DDQNLightning
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
from icecream import ic

checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="total_reward",
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

model = DDQNLightning(
    batch_size=256,
    warm_start_steps=7000,
    episode_length=5000,
    replay_size=10000,
    eps_min=0.01,
)
# ic(model)

now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(name=f"qMario-{now_dt}")
wandb.watch(model)

wandb_logger = WandbLogger()

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs=400000,
    logger=wandb_logger,
    gradient_clip_val= 30.0,
    # val_check_interval=50,
    # auto_lr_find=True,
    callbacks=[checkpoint_callback],
)

trainer.fit(model)
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
    monitor="global_step",
    mode="max",
    dirpath="model/",
    filename="sample-qmario-{epoch:02d}",
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
    warm_start_steps=4000,
    episode_length=4000,
    replay_size=4000,
)
# ic(model)

now_dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(name=f"qMario-{now_dt}")
wandb.watch(model)

trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    max_epochs=4000000,
    # val_check_interval=50,
    # auto_lr_find=True,
    callbacks=[checkpoint_callback],
)

trainer.fit(model)
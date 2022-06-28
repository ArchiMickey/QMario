import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from core.distd3qn import DistD3QNLightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from datetime import datetime

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="last_test_reward",
    mode="max",
    dirpath="model/",
    filename="qmario-{epoch:02d}",
    every_n_train_steps=1000,
)

Model = DistD3QNLightning(
    batch_size=512,
    lr=6.25e-5,
    min_lr=1e-6,
    gamma=0.9,
    target_update=10000,
    memory_size=100000,
    warm_start_size=10000,
    eps_start=1,
    eps_decay=0.9999,
    min_eps=0.02,
    alpha=0.5,
    beta=0.4,
    v_min=-15,
    v_max=15,
    atom_size=51,
    n_step=3,
    save_video=True,
    fps=24,
    video_rate=20,
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
    callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'), RichProgressBar()],
)

# trainer.tune(Model)
trainer.fit(Model)
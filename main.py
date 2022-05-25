import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from core.model import DDQNLightning
from pytorch_lightning.callbacks import ModelCheckpoint


checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="global_step",
    mode="max",
    dirpath="model/",
    filename="sample-qmario-{epoch:02d}",
    every_n_train_steps=5e5,
)

model = DDQNLightning()
wandb_logger = WandbLogger(name="qMario")
trainer = pl.Trainer(
    accelerator="gpu",
    devices = 1 if torch.cuda.is_available() else None,
    logger=wandb_logger,
    max_epochs=40000,
    # val_check_interval=50,
    auto_lr_find=True,
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
)

trainer.tune(model)
trainer.fit(model)
from core.model import DQNLightning
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

AVAIL_GPUS = min(1, torch.cuda.device_count())
model = DQNLightning()
wandb_logger = WandbLogger()


trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=40000,
    val_check_interval=100,
    gradient_clip_val=1.0,
    logger=wandb_logger
)
# trainer.tune(model)
trainer.fit(model)
trainer.save_checkpoint("Mario.ckpt")
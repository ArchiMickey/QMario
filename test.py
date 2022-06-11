import pytorch_lightning as pl

from core.model import DDQNLightning

model = DDQNLightning(
    batch_size=256,
    warm_start_size=300,
    episode_length=300,
    replay_size=10000,
    eps_min=0.01,
    save_video=True,
    fps=24,
)

trainer = pl.Trainer(
    max_steps=1,
    accelerator="gpu",
)
trainer.test(model=model, ckpt_path="model/qmario-epoch=40312.ckpt")
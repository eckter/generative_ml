from dataclasses import dataclass, field

import torch


@dataclass()
class HyperParams:
    n_epoch: int = field(default=5)
    learning_rate: float = field(default=1e-5)
    batch_size: int = field(default=32)
    device: str = field(default="cpu")
    latent_size: int = field(default=16)
    image_size: int = field(default=64)
    sample_every_n_step: int = field(default=10)

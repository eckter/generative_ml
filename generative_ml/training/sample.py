from dataclasses import dataclass

import torch


@dataclass
class Sample:
    inputs: torch.tensor
    outputs: torch.tensor
    types: torch.tensor
    n_step: int

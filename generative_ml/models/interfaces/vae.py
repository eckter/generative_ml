from abc import ABC, abstractmethod

import torch


class VAE(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplemented()

    @abstractmethod
    def encode(self, x: torch.tensor) -> torch.tensor:
        raise NotImplemented()

    @abstractmethod
    def decode(self, x: torch.tensor) -> torch.tensor:
        raise NotImplemented()

    @abstractmethod
    def get_kl(self) -> torch.tensor:
        raise NotImplemented()

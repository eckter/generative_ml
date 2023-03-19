import torch

from generative_ml.models.dummy_vae.decoder import DummyDecoder
from generative_ml.models.dummy_vae.variational_encoder import DummyVE
from generative_ml.models.interfaces.vae import VAE


class DummyVAE(VAE):
    def __init__(self, latent_size: int, img_size: int):
        super(DummyVAE, self).__init__()

        self.encoder = DummyVE(latent_size)
        self.decoder = DummyDecoder(latent_size, img_size)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def get_kl(self) -> torch.tensor:
        return self.encoder.kl

    def to(self, device):
        self.encoder.normal.loc = self.encoder.normal.loc.to(device)
        self.encoder.normal.scale = self.encoder.normal.scale.to(device)
        return super(DummyVAE, self).to(device)

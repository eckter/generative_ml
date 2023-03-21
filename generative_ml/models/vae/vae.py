import torch

from generative_ml.models.interfaces import VAEInterface
from generative_ml.models.vae.decoder import VAEDecoder
from generative_ml.models.vae.variational_encoder import VEncoder


class VAE(VAEInterface):
    def __init__(self, latent_size: int, img_size: int):
        super(VAEInterface, self).__init__()

        self.encoder = VEncoder(latent_size)
        self.decoder = VAEDecoder(latent_size, img_size)

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
        return super(VAE, self).to(device)

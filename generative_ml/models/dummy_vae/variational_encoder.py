import torch


class DummyVE(torch.nn.Module):
    def __init__(self, latent_size: int):
        super(DummyVE, self).__init__()

        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(3, latent_size, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh()
        ])

        self.mu_layer = torch.nn.Linear(latent_size, latent_size)
        self.sigma_layer = torch.nn.Linear(latent_size, latent_size)
        self.normal = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.mean(-1).mean(-1)

        mu = self.mu_layer(x)
        sigma = torch.exp(self.sigma_layer(x))

        res = mu + sigma * self.normal.sample(mu.shape)

        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        return res

import torch


class VEncoder(torch.nn.Module):
    def __init__(self, latent_size: int):
        super(VEncoder, self).__init__()

        def convolution(in_depth, out_depth, kernel=3, stride=1, padding=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_depth, out_depth, kernel_size=kernel, stride=stride, padding=padding),
                torch.nn.MaxPool2d(2),
                torch.nn.LeakyReLU(0.2)
            )

        self.convs = torch.nn.ModuleList([
            convolution(3, 16),
            convolution(16, 32),
            convolution(32, 32),
            convolution(32, latent_size),
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

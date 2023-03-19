import torch


class DummyDecoder(torch.nn.Module):
    def __init__(self, latent_size: int, img_size: int):
        super(DummyDecoder, self).__init__()

        intermediate_depth = 16

        def deconvolution(in_depth, out_depth, kernel=2, stride=2, padding=0):
            return torch.nn.ModuleList([
                torch.nn.ConvTranspose2d(in_depth, out_depth, kernel_size=kernel, stride=stride, padding=padding),
                torch.nn.Tanh()
            ])
        self.layers = torch.nn.ModuleList()
        self.layers += deconvolution(latent_size, intermediate_depth, kernel=8, stride=1, padding=0)
        size = 8
        while size != img_size:
            self.layers += deconvolution(intermediate_depth, intermediate_depth)
            size *= 2
            assert size <= img_size

        self.layers += deconvolution(intermediate_depth, 3, kernel=3, stride=1, padding=1)

    def forward(self, latent):
        x = latent.reshape(latent.shape[0], -1, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return x


import torch
from torch.utils import data
from generative_ml import HyperParams
from generative_ml.models.interfaces import VAEInterface
from generative_ml.training.sample import Sample


def train_vae(
        model: VAEInterface,
        training_data: data.Dataset,
        hyperparams: HyperParams
):
    device = torch.device(hyperparams.device)
    model = model.to(device)
    loader = data.DataLoader(
        dataset=training_data,
        batch_size=hyperparams.batch_size,
        shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)

    i_step = 0
    for epoch in range(hyperparams.n_epoch):
        for batch, (images, types) in enumerate(loader):
            model.train()
            images = images.to(device)
            optimizer.zero_grad()

            images_hat = model(images)
            loss = ((images - images_hat) ** 2).sum() + model.get_kl()
            loss.backward()
            optimizer.step()
            if i_step % hyperparams.sample_every_n_step == 0:
                yield Sample(
                    inputs=images,
                    outputs=images_hat.detach(),
                    types=types,
                    n_step=i_step
                )
            i_step += 1


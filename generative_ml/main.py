import click
from pathlib import Path

from generative_ml import training, PokeDataset, HyperParams
from generative_ml.models.dummy_vae import DummyVAE


DEFAULT_DATA = Path(__file__).parents[1] / "data"


@click.group()
def cli():
    pass


@cli.command()
@click.option('--data-dir', '-d', type=click.Path(path_type=Path), default=DEFAULT_DATA)
def train_vae(data_dir):
    params = HyperParams()
    data = PokeDataset(data_dir, image_size=params.image_size, n_pokemon=5)
    model = DummyVAE(params.latent_size, params.image_size)
    for sample in training.train_vae(model, data, params):
        print(sample.n_step)


if __name__ == "__main__":
    cli()

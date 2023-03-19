import click


@click.group()
def cli():
    pass


@cli.command()
def train():
    print("hello")

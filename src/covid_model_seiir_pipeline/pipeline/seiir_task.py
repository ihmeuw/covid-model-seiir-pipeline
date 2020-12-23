import click

from .regression.task import beta_regression


@click.group()
def stask():
    """Parent command for individual tasks."""
    pass


stask.add_command(beta_regression, name='beta_regression')



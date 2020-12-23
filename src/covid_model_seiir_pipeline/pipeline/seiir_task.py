import click

from .regression.task import beta_regression


@click.group()
def seiir_task():
    """Parent command for individual tasks."""
    pass


seiir_task.add_command(beta_regression)

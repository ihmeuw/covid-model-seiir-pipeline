import click

from .regression.task import beta_regression
from .forecasting.task import (
    beta_residual_scaling,
    beta_forecast,
)


@click.group()
def stask():
    """Parent command for individual tasks."""
    pass


stask.add_command(beta_regression, name='beta_regression')
stask.add_command(beta_residual_scaling, name='beta_residual_scaling')
stask.add_command(beta_forecast, name='beta_forecast')

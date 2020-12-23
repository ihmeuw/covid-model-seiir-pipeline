import click

from .regression import (
    REGRESSION_JOBS,
    beta_regression
)
from .forecasting import (
    FORECAST_JOBS,
    beta_residual_scaling,
    beta_forecast,
)
from .postprocessing import (
    POSTPROCESSING_JOBS,
    resample_map,
    postprocess,
)


@click.group()
def stask():
    """Parent command for individual seiir pipeline tasks."""
    pass


stask.add_command(beta_regression, name=REGRESSION_JOBS.regression)
stask.add_command(beta_residual_scaling, name=FORECAST_JOBS.scaling)
stask.add_command(beta_forecast, name=FORECAST_JOBS.forecast)
stask.add_command(resample_map, name=POSTPROCESSING_JOBS.resample)
stask.add_command(postprocess, name=POSTPROCESSING_JOBS.postprocess)

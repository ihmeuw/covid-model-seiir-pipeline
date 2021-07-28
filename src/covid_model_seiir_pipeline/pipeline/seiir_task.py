import click

from covid_model_seiir_pipeline.pipeline.fit_oos import (
    FIT_JOBS,
    beta_fit,
)
from covid_model_seiir_pipeline.pipeline.regression import (
    REGRESSION_JOBS,
    beta_regression,
    hospital_correction_factors,
)
from covid_model_seiir_pipeline.pipeline.forecasting import (
    FORECAST_JOBS,
    beta_residual_scaling,
    beta_forecast,
)
from covid_model_seiir_pipeline.pipeline.counterfactual import (
    COUNTERFACTUAL_JOBS,
    counterfactual,
)
from covid_model_seiir_pipeline.pipeline.postprocessing import (
    POSTPROCESSING_JOBS,
    resample_map,
    postprocess,
)
from covid_model_seiir_pipeline.pipeline.diagnostics import (
    DIAGNOSTICS_JOBS,
    grid_plots,
)


@click.group()
def stask():
    """Parent command for individual seiir pipeline tasks."""
    pass


stask.add_command(beta_fit, name=FIT_JOBS.fit)
stask.add_command(beta_regression, name=REGRESSION_JOBS.regression)
stask.add_command(hospital_correction_factors, name=REGRESSION_JOBS.hospital_correction_factors)
stask.add_command(beta_residual_scaling, name=FORECAST_JOBS.scaling)
stask.add_command(beta_forecast, name=FORECAST_JOBS.forecast)
stask.add_command(counterfactual, name=COUNTERFACTUAL_JOBS.counterfactual)
stask.add_command(resample_map, name=POSTPROCESSING_JOBS.resample)
stask.add_command(postprocess, name=POSTPROCESSING_JOBS.postprocess)
stask.add_command(grid_plots, name=DIAGNOSTICS_JOBS.grid_plots)

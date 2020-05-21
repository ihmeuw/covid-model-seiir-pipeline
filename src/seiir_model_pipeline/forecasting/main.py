from pathlib import Path

from covid_shared import cli_tools

from seiir_model_pipeline.forecasting import ForecastSpecification


def do_beta_forecast(app_metadata: cli_tools.Metadata,
                     forecast_specification: ForecastSpecification,
                     regression_root: Path,
                     output_root: Path):
    pass

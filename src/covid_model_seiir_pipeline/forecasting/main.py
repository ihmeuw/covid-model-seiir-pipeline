from pathlib import Path

from covid_shared import cli_tools

from covid_model_seiir_pipeline.forecasting import ForecastSpecification


def do_beta_forecast(app_metadata: cli_tools.Metadata,
                     forecast_specification: ForecastSpecification,
                     regression_root: Path,
                     output_root: Path):
    pass

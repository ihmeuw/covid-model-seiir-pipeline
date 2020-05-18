from pathlib import Path

from covid_shared import cli_tools


def do_beta_forecast(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification,
                       infection_root: Path,
                       covariates_root: Path,
                       output_root: Path):

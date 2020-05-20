from pathlib import Path

from covid_shared import cli_tools

from seiir_model_pipeline.regression.specification import RegressionSpecification, validate_specification


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification,
                       ode_fit_root: Path,
                       covariates_root: Path,
                       output_root: Path):
    validate_specification(regression_specification)
    pass

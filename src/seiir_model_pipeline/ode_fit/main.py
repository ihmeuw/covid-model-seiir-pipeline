from pathlib import Path

from covid_shared import cli_tools

from seiir_model_pipeline.ode_fit import FitSpecification


def do_beta_fit(app_metadata: cli_tools.Metadata,
                regression_specification: FitSpecification,
                infection_root: Path,
                output_root: Path):
    pass

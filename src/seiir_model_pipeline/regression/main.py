from pathlib import Path

from covid_shared import cli_tools

from seiir_model_pipeline.exceptions import VersionAlreadyExists
from seiir_model_pipeline.regression.specification import (RegressionSpecification,
                                                           validate_specification)
from seiir_model_pipeline.regression.version import RegressionVersion


# TODO: add to cli
resume = False


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification,
                       output_dir: Path):
    validate_specification(regression_specification)

    try:
        regress_version = RegressionVersion.init_version(
            regression_specification=regression_specification,
            regression_dir=output_dir
        )
    except VersionAlreadyExists:
        if resume:
            regress_version = RegressionVersion(
                regression_specification=regression_specification,
                regression_dir=output_dir
            )
        else:
            raise

    regress_version.run()

from argparse import ArgumentParser, Namespace
import shlex
from typing import Optional

from covid_shared.cli_tools import Metadata
from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger

from covid_model_seiir_pipeline.pipeline import (
    RegressionSpecification,
    do_beta_regression,
)


def run_oos_regression(regression_specification_path: str) -> None:
    regression_specification = RegressionSpecification.from_path(regression_specification_path)
    do_beta_regression(Metadata(), regression_specification, preprocess_only=False)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--regression-specification-path", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)
    args = parse_arguments()
    run_oos_regression(args.regression_specification_path)


if __name__ == '__main__':
    main()

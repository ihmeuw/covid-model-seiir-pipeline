from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
import shutil
from typing import Optional

from covid_shared.cli_tools import Metadata
from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger

from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.main import do_beta_forecast


def run_oos_forecast(forecast_specification_path: str) -> None:
    forecast_specification = ForecastSpecification.from_path(forecast_specification_path)
    do_beta_forecast(Metadata(), forecast_specification, preprocess_only=False)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-specification-path", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)
    args = parse_arguments()
    run_oos_forecast(args.forecast_specification_path)


if __name__ == '__main__':
    main()

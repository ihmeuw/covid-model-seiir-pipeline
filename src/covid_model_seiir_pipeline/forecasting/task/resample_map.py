from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional

import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting import postprocessing_lib as pp

log = logging.getLogger(__name__)


def run_resample_map(forecast_version: str) -> None:
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    resampling_params = forecast_spec.resampling
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    deaths, *_ = pp.load_output_data(resampling_params.reference_scenario, data_interface)
    deaths = pd.concat(deaths, axis=1)
    resampling_map = pp.build_resampling_map(deaths, resampling_params)
    data_interface.save_resampling_map(resampling_map)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    run_resample_map(forecast_version=args.forecast_version)


if __name__ == '__main__':
    main()

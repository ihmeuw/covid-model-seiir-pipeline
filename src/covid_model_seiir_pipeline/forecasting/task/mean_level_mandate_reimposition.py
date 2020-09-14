from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import shlex

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface


def run_mean_level_mandate_reimposition(forecast_version: str, scenario_name: str, reimposition_number: int):
    logger.info(f"Initiating SEIIR mean mean level mandate reimposition {reimposition_number} "
                f"for scenario {scenario_name}.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    raise NotImplementedError


def parse_arguments(arg_str: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)
    parser.add_argument("--reimposition-number", type=int, required=True)

    if arg_str is not None:
        arg_list = shlex.split(arg_str)
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)  # Debug level
    args = parse_arguments()
    run_mean_level_mandate_reimposition(forecast_version=args.forecast_version,
                                        scenario_name=args.scenario_name,
                                        reimposition_number=args.reimposition_number)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import shlex

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting import postprocessing_lib as pp
from covid_model_seiir_pipeline.forecasting import model


def run_mean_level_mandate_reimposition(forecast_version: str, scenario_name: str, reimposition_number: int):
    logger.info(f"Initiating SEIIR mean mean level mandate reimposition {reimposition_number} "
                f"for scenario {scenario_name}.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    resampling_map = data_interface.load_resampling_map()
    deaths = pp.load_deaths(scenario_name, data_interface)
    deaths = pd.concat(deaths, axis=1)
    deaths = pp.resample_draws(deaths, resampling_map)
    deaths = pp.summarize(deaths)
    deaths = deaths['mean'].rename('deaths').reset_index()
    deaths['date'] = pd.to_datetime(deaths['date'])

    modeled_locations = deaths['location_id'].unique().tolist()
    deaths = deaths.set_index(['location_id', 'date'])

    population = pp.load_populations(data_interface)
    population = population[population.location_id.isin(modeled_locations) 
                            & (population.age_group_id == 22) 
                            & (population.sex_id == 3)].set_index('location_id')['population']

    min_wait, days_on, reimposition_threshold = model.unpack_parameters(scenario_spec.algorithm_params)

    previous_dates = pd.Series(pd.NaT, index=population.index)
    for previous_reimposition in range(reimposition_number-1, 0, -1):
        these_dates = data_interface.load_reimposition_dates(scenario=scenario_name,
                                                             reimposition_number=previous_reimposition)
        these_dates = pd.to_datetime(these_dates.set_index('location_id')['reimposition_date'])
        these_dates = these_dates.reindex(previous_dates.index)
        this_reimposition = previous_dates.isnull() & these_dates.notnull()
        previous_dates.loc[this_reimposition] = these_dates.loc[this_reimposition]
    last_reimposition_end_date = previous_dates + days_on
    reimposition_date = model.compute_reimposition_date(deaths, population, reimposition_threshold,
                                                        min_wait, last_reimposition_end_date)
    data_interface.save_reimposition_dates(reimposition_date.reset_index(), scenario=scenario_name,
                                           reimposition_number=reimposition_number)


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

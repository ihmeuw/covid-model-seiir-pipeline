from argparse import ArgumentParser, Namespace
import functools
import logging
import multiprocessing
from pathlib import Path
import shlex
from typing import Optional

import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface

log = logging.getLogger(__name__)


def run_seir_postprocessing(forecast_version: str, scenario: str) -> None:
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    resampling_map = data_interface.load_resampling_map()

    deaths = data_interface.load_concatenated_outputs(scenario, 'deaths')
    infections = data_interface.load_concatenated_outputs(scenario, 'infections')
    r_effective = data_interface.load_concatenated_outputs(scenario, 'r_effective')
    betas = data_interface.load_concatenated_outputs(scenario, 'betas')
    beta_residuals = data_interface.load_concatenated_outputs(scenario, 'beta_residuals')

    deaths, infections, r_effective, betas, beta_residuals = resample_draws(resampling_map,
                                                                            deaths, infections, r_effective,
                                                                            betas, beta_residuals)
    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    cumulative_infections = infections.groupby(level='location_id').cumsum()
    output_draws = {
        'daily_deaths': deaths,
        'daily_infections': infections,
        'r_effective': r_effective,
        'cumulative_deaths': cumulative_deaths,
        'cumulative_infections': cumulative_infections,
        'betas': betas,
        'log_beta_residuals': beta_residuals,
    }
    for measure, data in output_draws.items():
        data_interface.save_output_draws(data.reset_index(), scenario, measure)
        summarized_data = summarize(data)
        data_interface.save_output_summaries(summarized_data.reset_index(), scenario, measure)


def summarize(data: pd.DataFrame):
    data['mean'] = data.mean(axis=1)
    data['upper'] = data.quantile(.975, axis=1)
    data['lower'] = data.quantile(.025, axis=1)
    return data[['mean', 'upper', 'lower']]


def resample_draws(resampling_map, *measure_data: pd.DataFrame):
    _runner = functools.partial(
        resample_draws_by_measure,
        resampling_map=resampling_map
    )
    with multiprocessing.Pool(len(measure_data)) as pool:
        resampled = pool.map(_runner, measure_data)
    return resampled


def resample_draws_by_measure(measure_data: pd.DataFrame, resampling_map):
    output = []
    for location_id, (to_drop, to_fill) in resampling_map.items():
        loc_data = measure_data.loc[location_id]
        loc_data[to_drop] = loc_data[to_fill]
        loc_data = pd.concat({location_id: loc_data}, names=['location_id'])
        output.append(loc_data)

    resampled = pd.concat(output)
    resampled.columns = [f'draw_{draw}' for draw in resampled.columns]
    return resampled


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
    run_seir_postprocessing(forecast_version=args.forecast_version)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser, Namespace
import functools
import logging
import multiprocessing
from pathlib import Path
import shlex
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface

log = logging.getLogger(__name__)

LOAD_OUTPUT_CPUS = 3

# TODO: reevaluate how multiprocessing is being used. Still firmly in the
#   "make it work" phase


def run_seir_postprocessing(forecast_version: str) -> None:
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    beta_residuals = load_beta_residuals(data_interface)

    resampling_params = forecast_spec.postprocessing.resampling
    resampling_map = build_resampling_map(resampling_params, data_interface)

    for scenario in forecast_spec.scenarios:
        deaths, infections, r_effective = load_output_data(scenario, data_interface)
        deaths, infections, r_effective = concat_measures(deaths, infections, r_effective)
        deaths, infections, r_effective = resample_draws(resampling_map, deaths, infections, r_effective)
        cumulative_deaths = deaths.groupby(level='location_id').cumsum()
        cumulative_infections = infections.groupby(level='location_id').cumsum()
        betas = load_betas(scenario, data_interface)
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


def load_betas(scenario: str, data_interface: ForecastDataInterface):
    betas = []
    for draw_id in range(data_interface.get_n_draws()):
        components = data_interface.load_components(scenario, draw_id)
        components['date'] = pd.to_datetime(components['date'])
        draw_betas = (components
                      .set_index(['location_id', 'date'])
                      .sort_index()['beta']
                      .rename(f'draw_{draw_id}'))
        betas.append(draw_betas)
    return pd.concat(betas, axis=1)


def load_beta_residuals(data_interface: ForecastDataInterface):
    beta_residuals = []
    for draw_id in range(data_interface.get_n_draws()):
        beta_regression = data_interface.load_beta_regression(draw_id)
        beta_regression = (beta_regression
                           .set_index(['location_id', 'date'])
                           .sort_index()[['beta', 'beta_pred']])
        beta_residual = np.log(beta_regression['beta'] / beta_regression['beta_pred']).rename(f'draw_{draw_id}')
        beta_residuals.append(beta_residual)

    return pd.concat(beta_residuals, axis=1)


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


def build_resampling_map(resampling_params: Dict, data_interface: ForecastDataInterface):
    resampling_ref_scenario = resampling_params['reference_scenario']
    deaths, *_ = load_output_data(resampling_ref_scenario, data_interface)
    deaths = concat_measures(deaths)[0]
    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    max_deaths = cumulative_deaths.groupby(level='location_id').max()
    upper_deaths = max_deaths.quantile(resampling_params['upper_quantile'], axis=1)
    lower_deaths = max_deaths.quantile(resampling_params['lower_quantile'], axis=1)
    resample_map = {}
    for location_id in max_deaths.index:
        upper, lower = upper_deaths.at[location_id], lower_deaths.at[location_id]
        loc_deaths = max_deaths.loc[location_id]
        to_resample = loc_deaths[(upper < loc_deaths) | (loc_deaths < lower)].index.tolist()
        to_fill = np.random.choice(loc_deaths.index, len(to_resample), replace=False)
        resample_map[location_id] = (to_resample, to_fill)
    return resample_map


def load_output_data(scenario: str, data_interface: ForecastDataInterface):
    _runner = functools.partial(
        load_output_data_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(LOAD_OUTPUT_CPUS) as pool:
        outputs = list(pool.map(_runner, draws))
    deaths, infections, r_effective = zip(*outputs)

    return deaths, infections, r_effective


def load_output_data_by_draw(draw_id: int, scenario: str, data_interface: ForecastDataInterface):
    draw_df = data_interface.load_raw_outputs(scenario, draw_id)
    draw_df = draw_df.set_index(['location_id', 'date']).sort_index()
    deaths = draw_df['deaths'].rename(draw_id)
    infections = draw_df['infections'].rename(draw_id)
    r_effective = draw_df['r_effective'].rename(draw_id)
    return deaths, infections, r_effective


def concat_measures(*measure_data: List[pd.Series]) -> List[pd.DataFrame]:
    _runner = functools.partial(
        pd.concat,
        axis=1
    )
    with multiprocessing.Pool(LOAD_OUTPUT_CPUS) as pool:
        measure_data = pool.map(_runner, measure_data)
    return measure_data


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

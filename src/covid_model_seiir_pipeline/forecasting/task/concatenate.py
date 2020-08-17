from argparse import ArgumentParser, Namespace
import functools
import logging
import multiprocessing
from pathlib import Path
import shlex
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import FORECAST_SCALING_CORES

log = logging.getLogger(__name__)


def run_concatenate(forecast_version: str, scenario_name: str) -> None:
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    deaths, infections, r_effective = load_output_data(scenario_name, data_interface)
    betas = load_betas(scenario_name, data_interface)
    beta_residuals = load_beta_residuals(data_interface)
    deaths, infections, r_effective, betas, beta_residuals = concat_measures(deaths, infections, r_effective,
                                                                             betas, beta_residuals)
    data_interface.save_concatenated_outputs(deaths, scenario=scenario_name, measure='deaths')
    data_interface.save_concatenated_outputs(infections, scenario=scenario_name, measure='infections')
    data_interface.save_concatenated_outputs(r_effective, scenario=scenario_name, measure='r_effective')
    data_interface.save_concatenated_outputs(betas, scenario=scenario_name, measure='betas')
    data_interface.save_concatenated_outputs(beta_residuals, scenario=scenario_name, measure='beta_residuals')

    resampling_params = forecast_spec.postprocessing.resampling
    if scenario_name == resampling_params['reference_scenario']:
        resampling_map = build_resampling_map(deaths, resampling_params)
        data_interface.save_resampling_map(resampling_map)


def build_resampling_map(deaths, resampling_params):
    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    max_deaths = cumulative_deaths.groupby(level='location_id').max()
    upper_deaths = max_deaths.quantile(resampling_params['upper_quantile'], axis=1)
    lower_deaths = max_deaths.quantile(resampling_params['lower_quantile'], axis=1)
    resample_map = {}
    for location_id in max_deaths.index:
        upper, lower = upper_deaths.at[location_id], lower_deaths.at[location_id]
        loc_deaths = max_deaths.loc[location_id]
        to_resample = loc_deaths[(upper < loc_deaths) | (loc_deaths < lower)].index.tolist()
        np.random.seed(12345)
        to_fill = np.random.choice(loc_deaths.index, len(to_resample), replace=False).tolist()
        resample_map[location_id] = {'to_resample': to_resample,
                                     'to_fill': to_fill}
    return resample_map


def load_output_data(scenario: str, data_interface: ForecastDataInterface):
    _runner = functools.partial(
        load_output_data_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)
    deaths, infections, r_effective = zip(*outputs)

    return deaths, infections, r_effective


def load_betas(scenario: str, data_interface: ForecastDataInterface) -> List[pd.Series]:
    betas = []
    for draw_id in range(data_interface.get_n_draws()):
        components = data_interface.load_components(scenario, draw_id)
        draw_betas = (components
                      .sort_index()['beta']
                      .rename(draw_id))
        betas.append(draw_betas)
    return betas


def load_beta_residuals(data_interface: ForecastDataInterface) -> List[pd.Series]:
    beta_residuals = []
    for draw_id in range(data_interface.get_n_draws()):
        beta_regression = data_interface.load_beta_regression(draw_id)
        beta_regression = (beta_regression
                           .set_index(['location_id', 'date'])
                           .sort_index()[['beta', 'beta_pred']])
        beta_residual = np.log(beta_regression['beta'] / beta_regression['beta_pred']).rename(draw_id)
        beta_residuals.append(beta_residual)
    return beta_residuals


def load_output_data_by_draw(draw_id: int, scenario: str,
                             data_interface: ForecastDataInterface) -> Tuple[pd.Series, pd.Series, pd.Series]:
    draw_df = data_interface.load_raw_outputs(scenario, draw_id)
    draw_df = draw_df.set_index(['location_id', 'date']).sort_index()
    deaths = draw_df['deaths'].rename(draw_id)
    infections = draw_df['infections'].rename(draw_id)
    r_effective = draw_df['r_effective'].rename(draw_id)
    return deaths, infections, r_effective


def concat_measures(*measure_data: List[pd.Series]) -> List[pd.DataFrame]:
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        output_data = pool.map(concat_draws, measure_data)
    return output_data


def concat_draws(measure_data: List[pd.Series]) -> pd.DataFrame:
    # 3x faster than pd.concat for reasons I don't understand.
    measure_data = functools.reduce(lambda a, b: pd.merge(a, b, left_index=True, right_index=True, how='outer'), measure_data)
    measure_data.columns = [f'draw_{i}' for i in measure_data.columns]
    return measure_data.reset_index()


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    run_concatenate(forecast_version=args.forecast_version,
                    scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()

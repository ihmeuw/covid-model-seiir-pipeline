import functools
import multiprocessing
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import FORECAST_SCALING_CORES


# TODO: make a model subpackage and put this there.

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


def load_output_data_by_draw(draw_id: int, scenario: str,
                             data_interface: ForecastDataInterface) -> Tuple[pd.Series, pd.Series, pd.Series]:
    draw_df = data_interface.load_raw_outputs(scenario, draw_id)
    draw_df = draw_df.set_index(['location_id', 'date']).sort_index()
    deaths = draw_df['deaths'].rename(draw_id)
    infections = draw_df['infections'].rename(draw_id)
    r_effective = draw_df['r_effective'].rename(draw_id)
    return deaths, infections, r_effective


def load_covariates(scenario: str, cov_order: Dict[str, List[str]],
                    data_interface: ForecastDataInterface) -> Dict[str, List[pd.Series]]:
    _runner = functools.partial(
        load_covariates_by_draw,
        scenario=scenario,
        cov_order=cov_order,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)

    cov_names = [*cov_order['time_varying'], *cov_order['non_time_varying']]
    covariates = dict(zip(cov_names, zip(*outputs)))
    return covariates


def load_covariates_by_draw(draw_id: int, scenario: str,
                            cov_order: Dict[str, List[str]],
                            data_interface: ForecastDataInterface) -> Tuple[pd.Series, ...]:
    covariate_df = data_interface.load_raw_covariates(scenario, draw_id)
    covariate_df = covariate_df.set_index(['location_id', 'date']).sort_index()
    covariate_grouped = covariate_df.groupby(level='location_id')

    time_varying = [covariate_df[col].rename(draw_id) for col in cov_order['time_varying']]
    non_time_varying = [covariate_grouped[col].max().rename(draw_id) for col in cov_order['non_time_varying']]

    return (*time_varying, *non_time_varying)


def load_betas(scenario: str, data_interface: ForecastDataInterface) -> List[pd.Series]:
    _runner = functools.partial(
        load_betas_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        betas = pool.map(_runner, draws)
    return betas


def load_betas_by_draw(draw_id: int, scenario: str, data_interface: ForecastDataInterface) -> pd.Series:
    components = data_interface.load_components(scenario, draw_id)
    draw_betas = (components
                  .sort_index()['beta']
                  .rename(draw_id))
    return draw_betas


def load_beta_residuals(data_interface: ForecastDataInterface) -> List[pd.Series]:
    _runner = functools.partial(
        load_beta_residuals_by_draw,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        beta_residuals = pool.map(_runner, draws)
    return beta_residuals


def load_beta_residuals_by_draw(draw_id: int, data_interface: ForecastDataInterface) -> pd.Series:
    beta_regression = data_interface.load_beta_regression(draw_id)
    beta_regression = (beta_regression
                       .set_index(['location_id', 'date'])
                       .sort_index()[['beta', 'beta_pred']])
    beta_residual = np.log(beta_regression['beta'] / beta_regression['beta_pred']).rename(draw_id)
    return beta_residual


def concat_draws(measure_data: List[pd.Series]) -> pd.DataFrame:
    # 3x faster than pd.concat for reasons I don't understand.
    measure_data = functools.reduce(lambda a, b: pd.merge(a, b, left_index=True, right_index=True, how='outer'),
                                    measure_data)
    measure_data.columns = [f'draw_{i}' for i in measure_data.columns]
    return measure_data.reset_index()


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

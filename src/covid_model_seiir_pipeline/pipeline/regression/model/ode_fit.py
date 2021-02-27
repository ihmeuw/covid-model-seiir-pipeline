from typing import Dict

from loguru import logger
import numba
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.pipeline.regression.model.containers import (
    ODEParameters,
)


def sample_parameters(draw_id: int, regression_parameters: Dict) -> ODEParameters:
    np.random.seed(draw_id)
    return ODEParameters(
        alpha=np.random.uniform(*regression_parameters['alpha']),
        sigma=np.random.uniform(*regression_parameters['sigma']),
        gamma1=np.random.uniform(*regression_parameters['gamma1']),
        gamma2=np.random.uniform(*regression_parameters['gamma2']),
    )


def run_beta_fit(past_infections: pd.Series,
                 population: pd.Series,
                 ode_parameters: ODEParameters) -> pd.DataFrame:
    beta_fit_dfs = []
    location_ids = past_infections.reset_index().location_id.unique()
    for location_id in location_ids:
        beta_fit = run_loc_beta_fit(
            infections=past_infections.loc[location_id],
            total_population=population.loc[location_id],
            location_id=location_id,
            ode_parameters=ode_parameters,
        )
        beta_fit_dfs.append(beta_fit)
    beta_fit = pd.concat(beta_fit_dfs)
    return beta_fit


def run_loc_beta_fit(infections: pd.Series,
                     total_population: float,
                     location_id: int,
                     ode_parameters: ODEParameters) -> pd.DataFrame:
    infections = filter_to_epi_threshold(location_id, infections)

    date = pd.Series(infections.index.values)
    t = (date - date.min()).dt.days.values
    obs = infections.values

    initial_condition = np.array([
        total_population - obs[0] - (obs[0] / 5) ** (1.0 / ode_parameters.alpha),  # S
        obs[0],                                                                    # E
        (obs[0] / 5) ** (1.0 / ode_parameters.alpha),                              # I1
        0,                                                                         # I2
        0,                                                                         # R
    ])
    parameters = np.vstack([
        obs,
        [ode_parameters.sigma] * len(obs),
        [ode_parameters.gamma1] * len(obs),
        [ode_parameters.gamma2] * len(obs),
    ])

    result = math.solve_ode(
        system=past_system,
        t=t,
        init_cond=initial_condition,
        params=parameters,
    )
    components = pd.DataFrame(
        data=result,
        columns=['S', 'E', 'I1', 'I2', 'R']
    )
    components['date'] = date
    components['location_id'] = location_id
    disease_density = components['S'] * (components['I1'] + components['I2'])**ode_parameters.alpha / total_population
    components['beta'] = obs / disease_density

    assert (components['S'] >= 0.0).all()

    return components


@numba.njit
def linear_first_order(_: float, y: np.ndarray, p: np.ndarray):
    c, f = p
    x = y[0]
    dx = -c * x + f
    return np.array([dx])


@numba.njit
def past_system(_: float, y: np.ndarray, p: np.ndarray):
    s, e, i1, i2, r = y
    new_e, sigma, gamma1, gamma2 = p

    ds = -new_e
    de = new_e - sigma*e
    di1 = sigma*e - gamma1*i1
    di2 = gamma1*i1 - gamma2*i2
    dr = gamma2*i2

    return np.array([
        ds, de, di1, di2, dr,
    ])


def filter_to_epi_threshold(location_id: int,
                            infections: pd.Series,
                            threshold: float = 50.) -> pd.Series:
    # noinspection PyTypeChecker
    start_date = infections.loc[threshold <= infections].index.min()
    while infections.loc[start_date:].count() <= 2:
        threshold *= 0.5
        logger.debug(f'Reduce infections threshold to {threshold} for location {location_id}.')
        # noinspection PyTypeChecker
        start_date = infections.loc[threshold <= infections].index.min()
        if threshold < 1e-6:
            start_date = infections.index.min()
            break
    return infections.loc[start_date:]

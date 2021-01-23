from typing import Dict, List

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
        day_shift=int(np.random.uniform(*regression_parameters['day_shift'])),
        solver_dt=regression_parameters['solver_dt'],
    )


def run_beta_fit(past_infections: pd.Series,
                 population: pd.Series,
                 location_ids: List[int],
                 ode_parameters: ODEParameters) -> pd.DataFrame:
    beta_fit_dfs = []
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
    today = infections.index.max()
    end_date = today - pd.Timedelta(days=ode_parameters.day_shift)
    infections = filter_to_epi_threshold(location_id, infections, end_date)

    date = pd.Series(infections.index.values)
    t = (date - date.min()).dt.days.values
    obs = infections.values

    shared_options = {'system': linear_first_order, 't': t, 'dt': ode_parameters.solver_dt}

    susceptible = math.solve_ode(
        init_cond=np.array([total_population - obs[0] - (obs[0] / 5.0) ** (1.0 / ode_parameters.alpha)]),
        params=np.vstack([
            [0.] * len(t),
            -obs,
        ]),
        **shared_options,
    ).ravel()

    exposed = math.solve_ode(
        init_cond=np.array([obs[0]]),
        params=np.vstack([
            [ode_parameters.sigma] * len(t),
            obs,
        ]),
        **shared_options,
    ).ravel()

    infectious_1 = math.solve_ode(
        init_cond=np.array([(obs[0] / 5.0) ** (1.0 / ode_parameters.alpha)]),
        params=np.vstack([
            [ode_parameters.gamma1] * len(t),
            ode_parameters.sigma * exposed,
        ]),
        **shared_options,
    ).ravel()

    infectious_2 = math.solve_ode(
        init_cond=np.array([0.]),
        params=np.vstack([
            [ode_parameters.gamma2] * len(t),
            ode_parameters.gamma1 * infectious_1,
        ]),
        **shared_options,
    ).ravel()

    removed = math.solve_ode(
        init_cond=np.array([0.]),
        params=np.vstack([
            [0.] * len(t),
            ode_parameters.gamma2 * infectious_2,
        ]),
        **shared_options
    ).ravel()

    components = {
        'S': susceptible,
        'E': exposed,
        'I1': infectious_1,
        'I2': infectious_2,
        'R': removed,
    }

    neg_susceptible_idx = susceptible < 0.0
    if np.any(neg_susceptible_idx):
        id_min = np.min(np.arange(susceptible.size)[neg_susceptible_idx])
        for c in components:
            components[c][id_min:] = components[c][id_min - 1]

    # get beta
    infectious = infectious_1 + infectious_2
    disease_density = susceptible * infectious**ode_parameters.alpha / total_population
    return pd.DataFrame({
        'location_id': location_id,
        'date': date,
        'beta': obs / disease_density,
        **components
    })


@numba.njit
def linear_first_order(t: float, y: np.ndarray, p: np.ndarray):
    c, f = p
    x = y[0]
    dx = -c * x + f
    return np.array([dx])


def filter_to_epi_threshold(location_id: int,
                            infections: pd.Series,
                            end_date: pd.Timestamp,
                            threshold: float = 50.) -> pd.Series:
    # noinspection PyTypeChecker
    start_date = infections.loc[threshold <= infections].index.min()
    while infections.loc[start_date:end_date].count() <= 2:
        threshold *= 0.5
        logger.debug(f'Reduce infections threshold to {threshold} for location {location_id}.')
        # noinspection PyTypeChecker
        start_date = infections.loc[threshold <= infections].index.min()
        if threshold < 1e-6:
            start_date = infections.index.min()
            break
    return infections.loc[start_date:end_date]

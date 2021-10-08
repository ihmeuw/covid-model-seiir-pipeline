import functools
import itertools
import multiprocessing
from typing import Tuple
import time

import numba
import numpy as np
import pandas as pd
import scipy.stats
import tqdm

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    PARAMETERS_NAMES,
    TRACKING_COMPARTMENTS,
    TRACKING_COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENT,
    VARIANT_GROUP,
)
from covid_model_seiir_pipeline.lib.ode_mk2.system import (
    system,
)


SOLVER_DT: float = 0.1


def run_ode_model(initial_condition: pd.DataFrame,
                  parameter_df: pd.DataFrame,
                  forecast: bool,
                  dt: float = SOLVER_DT, 
                  num_cores: int = 7):
    # Ensure data frame column labeling is consistent with expected index ordering.
    initial_condition = _sort_columns(initial_condition)
    location_ids = initial_condition.reset_index().location_id.unique().tolist()
    start = time.time()
    ics_and_params = [(location_id, initial_condition.loc[location_id], parameter_df.loc[location_id])
                      for location_id in location_ids]

    _runner = functools.partial(
        _run_loc_ode_model,
        dt=dt,
        forecast=forecast,
    )
    if num_cores == 1:
        compartments = [_runner(data) for data in tqdm.tqdm(ics_and_params)]
    else:
        with multiprocessing.Pool(num_cores) as pool:
            compartments = list(tqdm.tqdm(pool.imap(_runner, ics_and_params), total=len(ics_and_params)))

    compartments = pd.concat(compartments).reset_index().set_index(['location_id', 'date']).sort_index()
    print("Duration: ", time.time() - start, " seconds")
    return compartments


def _run_loc_ode_model(ic_and_params: Tuple[int, pd.DataFrame, pd.DataFrame],
                       dt: float,
                       forecast: bool):
    location_id, initial_condition, parameters = ic_and_params

    new_e_dates = initial_condition[initial_condition.filter(like='NewE').sum(axis=1) > 0].reset_index().date
    invasion_date = new_e_dates.min()
    ode_start_date = new_e_dates.max()

    t0 = (ode_start_date - invasion_date).days
    if forecast:
        assert t0 > 0
        ode_end_date = initial_condition.reset_index().date.max()

    else:
        assert t0 == 0
        ode_end_date = parameters.new_e.dropna().reset_index().date.max()
    tf = (ode_end_date - invasion_date)

    dates = initial_condition.reset_index().date
    t = (dates - invasion_date).dt.days.values
    y = initial_condition.to_numpy()
    params = parameters.to_numpy()
    # Split the daily "date" axis into chunks of width dt and interpolate
    # the compartments and parameters over the new time points.
    t_solve, y_solve, p_solve = _interpolate(t0, t, y, params, forecast, dt)

    # TODO: Replace with real distributions
    natural_dist = _sample_dist(_get_waning_dist(0, 90, 1500), t_solve) / dt
    vaccine_dist = _sample_dist(_get_waning_dist(0, 180, 3000), t_solve) / dt

    system = forecast_system if forecast else fit_system
    y_solve = _rk45_dde(
        system,
        t0, tf,
        t_solve,
        y_solve,
        p_solve,
        vaccine_dist,
        natural_dist,
        dt,
    )

    loc_compartments = pd.DataFrame(_uninterpolate(y_solve, t_solve, t),
                                    columns=initial_condition.columns,
                                    index=initial_condition.index)
    loc_compartments['location_id'] = location_id
    return loc_compartments


def _sort_columns(initial_condition: pd.DataFrame) -> pd.DataFrame:    
    initial_condition_columns = [f'{compartment}_{risk_group}'
                                 for risk_group, compartment
                                 in itertools.product(RISK_GROUP_NAMES, COMPARTMENTS_NAMES + TRACKING_COMPARTMENTS_NAMES)]
    assert set(initial_condition_columns) == set(initial_condition.columns)
    initial_condition = initial_condition.loc[:, initial_condition_columns]    

    return initial_condition


def _interpolate_y(t0: float,
                   t: np.ndarray,
                   t_solve: np.ndarray,
                   y: np.ndarray,
                   forecast: bool) -> np.ndarray:
    y_solve = np.empty((t_solve.size, y.shape[1]))
    for compartment in np.arange(y.shape[1]):
        if forecast:
            y_solve[:, compartment] = np.interp(t_solve, t, y[:, compartment])
            y_solve[t_solve > t0, compartment] = 0.
        else:
            y_solve[t_solve < t0, compartment] = y[0, compartment]
            y_solve[t_solve == t0, compartment] = y[t == t0, compartment]
            y_solve[t_solve > t0, compartment] = 0.
    return y_solve



def _interpolate(t: np.ndarray,                                  
                 t_solve: np.ndarray,
                 x: np.ndarray) -> np.ndarray:    
    x_solve = np.empty((t_solve.size, x.shape[1]))
    for param in np.arange(x.shape[1]):
        x_solve[:, param] = np.interp(t_solve, t, x[:, param])
    return x_solve



@numba.njit
def _rk45_dde(system,
              t0: float, tf: float,
              t_solve: np.ndarray,
              y_solve: np.ndarray,
              p_solve: np.ndarray,
              vaccine_dist: np.ndarray,
              natural_dist: np.ndarray,
              dt: float):
    num_time_points = t_solve.size
    system_size = TRACKING_COMPARTMENTS.max() + 1

    for time in np.arange(num_time_points):
        if not (t0 <= t_solve[time] <= tf):
            continue

        waned = _compute_waned_this_step(
            y_solve[:time],
            vaccine_dist[:time],
            natural_dist[:time],
            system_size,
        )

        k1 = system(
            t_solve[time - 1],
            y_solve[time - 1],
            waned,
            p_solve[2 * time - 2],
        )

        k2 = system(
            t_solve[time - 1] + dt / 2,
            y_solve[time - 1] + dt / 2 * k1,
            waned,
            p_solve[2 * time - 1],
        )

        k3 = system(
            t_solve[time - 1] + dt / 2,
            y_solve[time - 1] + dt / 2 * k2,
            waned,
            p_solve[2 * time - 1],
        )

        k4 = system(
            t_solve[time],
            y_solve[time - 1] + dt * k3,
            waned,
            p_solve[2 * time],
        )

        y_solve[time] = y_solve[time - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y_solve


@numba.njit
def _compute_waned_this_step(y_past: np.ndarray,
                             vaccine_dist: np.ndarray,
                             natural_dist: np.ndarray,
                             system_size: int) -> np.ndarray:
    waned = np.zeros(2)
    new_vax_immune_index = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, VARIANT_GROUP.total]
    new_r_index = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewR, VARIANT_GROUP.total]
    for i, (index, dist) in enumerate(((new_r_index, natural_dist), (new_vax_immune_index, vaccine_dist))):
        cumulative_total = y_past[:, index] + y_past[:, system_size + index]
        daily_total = cumulative_total[1:] - cumulative_total[:-1]
        waned[i] = (daily_total[::-1] * dist[:-1]).sum()
    return waned


@numba.njit
def _uninterpolate(y_solve: np.ndarray,
                   t_solve: np.ndarray,
                   t: np.ndarray):
    y_final = np.zeros((len(t), y_solve.shape[1]))
    for compartment in np.arange(y_solve.shape[1]):
        y_final[:, compartment] = np.interp(t, t_solve, y_solve[:, compartment])
    return y_final

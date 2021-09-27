import itertools
from typing import Tuple

import numba
import numpy as np
import pandas as pd
import scipy.stats

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    PARAMETERS_NAMES,
    TRACKING_COMPARTMENTS,
    TRACKING_COMPARTMENT,
    VARIANT_GROUP,
)
from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.system import (
    fit_system,
    forecast_system,
)


SOLVER_DT: float = 0.1


def run_ode_model(initial_condition: pd.DataFrame,
                  ode_parameters: Parameters,
                  forecast: bool,
                  dt: float = SOLVER_DT):
    # Ensure data frame column labeling is consistent with expected index ordering.
    initial_condition, parameter_df = _sort_columns(initial_condition, ode_parameters.to_df())
    # Convert into numpy arrays with meaningful dimensions that will work under an optimizer.
    # Dimensions are
    #
    # t0
    #    axis 0: location_id
    # t
    #    axis 0: date
    #    axis 1: location_id
    # y
    #    axis 0: date
    #    axis 1: location_id
    #    axis 2: compartment
    # params
    #    axis 0: date
    #    axis 1: location_id
    #    axis 2: parameter
    t0, t, y, params = _to_numpy(initial_condition, parameter_df, forecast)

    # Split the daily "date" axis into chunks of width dt and interpolate
    # the compartments and parameters over the new time points.
    t_solve, y_solve, p_solve = _interpolate(t0, t, y, params, forecast, dt)

    # TODO: Replace with real distributions
    natural_dist = _sample_dist(_get_waning_dist(0, 90, 1500), t_solve[:, 0]) / dt
    vaccine_dist = _sample_dist(_get_waning_dist(0, 180, 3000), t_solve[:, 0]) / dt

    system = forecast_system if forecast else fit_system
    y_solve = _rk45_dde(
        system,
        t0,
        t_solve,
        y_solve,
        p_solve,
        vaccine_dist,
        natural_dist,
        dt,
    )

    compartments = _uninterpolate(y_solve, t_solve, t)
    compartments = pd.DataFrame(compartments, columns=initial_condition.columns, index=initial_condition.index)
    return compartments


def _sort_columns(initial_condition: pd.DataFrame,
                  parameter_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    initial_condition_columns = [f'{compartment}_{risk_group}'
                                 for compartment, risk_group in itertools.product(COMPARTMENTS_NAMES, RISK_GROUP_NAMES)]
    assert set(initial_condition_columns) == set(initial_condition.columns)
    initial_condition = initial_condition.loc[:, initial_condition_columns]

    assert set(PARAMETERS_NAMES) == set(parameter_df.columns)
    parameter_df = parameter_df.loc[:, PARAMETERS_NAMES]

    return initial_condition, parameter_df


def _to_numpy(initial_condition: pd.DataFrame,
              parameter_df: pd.DataFrame,
              forecast: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    invasion_date, ode_start_date = _get_invasion_and_ode_start_dates_by_location(initial_condition)
    t0 = (ode_start_date - invasion_date).dt.days.values
    if forecast:
        assert np.all(t0 > 0)
    else:
        assert np.all(t0 == 0)

    dates_by_location = parameter_df.reset_index(level='date').date
    # parameter df is square by location date

    num_dates = len(dates_by_location.unique())
    location_ids = dates_by_location.index.unique().to_numpy()
    num_locs = len(location_ids)
    num_compartments = len(initial_condition.columns)
    num_params = len(parameter_df.columns)

    t = ((dates_by_location - invasion_date)
         .dt.days
         .values
         .reshape((num_locs, num_dates))).T

    y = (initial_condition
         .reorder_levels(['date', 'location_id'])
         .sort_index()
         .to_numpy()
         .reshape((num_dates, num_locs, num_compartments)))

    params = (parameter_df
              .reorder_levels(['date', 'location_id'])
              .sort_index()
              .to_numpy()
              .reshape((num_dates, num_locs, num_params)))

    return t0, t, y, params


def _get_invasion_and_ode_start_dates_by_location(initial_condition: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    dates = (initial_condition[initial_condition.filter(like='NewE').sum(axis=1) > 0]
             .reset_index(level='date')
             .date)
    invasion_date = dates.groupby('location_id').min()
    ode_start_date = dates.groupby('location_id').max()

    location_ids = initial_condition.reset_index().location_id.unique().values
    assert set(location_ids) == set(invasion_date.index)
    assert set(location_ids) == set(ode_start_date.index)
    return invasion_date, ode_start_date


@numba.njit
def _interpolate(t0: np.ndarray,
                 t: np.ndarray,
                 y: np.ndarray,
                 params: np.ndarray,
                 forecast: bool,
                 dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_locs = t.shape[1]
    num_time_points = np.arange(t[0, 0], t[-1, 0] + dt, dt).size
    num_compartments = y.shape[2]
    num_params = params.shape[2]

    t_solve = np.empty((num_time_points, num_locs))
    y_solve = np.empty((num_time_points, num_locs, num_compartments))
    # We'll need parameters on the half step
    p_solve = np.empty((2*num_time_points, num_locs, num_params))

    for location in np.arange(num_locs):
        t_solve[:, location] = np.round(np.arange(t[0, location], t[-1, location] + dt, dt), 2)
        t_params = np.round(np.arange(t[0, location], t[-1, location] + dt, dt/2), 2)

        for param in np.arange(num_params):
            p_solve[:, location, param] = np.interp(t_params, t[:, location], params[:, location, param])

        for compartment in np.arange(num_compartments):
            if forecast:
                y_solve = np.interp(t_solve[:, location], t[:, location], y[:, location, compartment])
                y_solve[t_solve[:, location] > t0[location], location, compartment] = 0.
            else:
                y_solve[t_solve[:, location] < t0[location], location, compartment] = (
                    y[0, location, compartment]
                )
                y_solve[t_solve[:, location] == t0[location], location, compartment] = (
                    y[t[:, location] == t0[location], location, compartment]
                )
                y_solve[t_solve[:, location] > t0[location], location, compartment] = (
                    0.
                )
    return t_solve, y_solve, p_solve


def _get_waning_dist(start, mean, var):
    return scipy.stats.gamma(loc=start, scale=var/mean, a=mean**2/var)


def _sample_dist(dist, t, rate=1):
    dist_cdf = dist.cdf(t[::rate])
    dist_discrete_pdf = dist_cdf[1:] - dist_cdf[:-1]
    dist_discrete_pdf = np.hstack([dist_discrete_pdf, dist_discrete_pdf[-1]])
    sampled_dist = np.zeros_like(t, dtype=dist_cdf.dtype)
    sampled_dist[::rate] = dist_discrete_pdf
    return sampled_dist


@numba.njit
def _rk45_dde(system,
              t0: np.ndarray,
              t_solve: np.ndarray,
              y_solve: np.ndarray,
              p_solve: np.ndarray,
              vaccine_dist: np.ndarray,
              natural_dist: np.ndarray,
              dt: float):
    num_time_points, num_locs = t_solve.shape
    system_size = TRACKING_COMPARTMENTS.max() + 1

    for time in np.arange(num_time_points):
        for location in np.arange(num_locs):
            # If we're not past t0, don't do anything yet.
            if t_solve[time, location] < t0[location]:
                continue

            if not time % 100:
                print("Percent complete: ", 100 * time / num_time_points, "%")

            waned = _compute_waned_this_step(
                y_solve[:time, location],
                vaccine_dist[:time],
                natural_dist[:time],
                system_size,
            )

            k1 = system(
                t_solve[time - 1, location],
                y_solve[time - 1, location],
                waned,
                p_solve[2 * time - 2, location],
            )

            k2 = system(
                t_solve[time - 1, location] + dt / 2,
                y_solve[time - 1, location] + dt / 2 * k1,
                waned,
                p_solve[2 * time - 1, location],
            )

            k3 = system(
                t_solve[time - 1, location] + dt / 2,
                y_solve[time - 1, location] + dt / 2 * k2,
                waned,
                p_solve[2 * time - 1, location],
            )

            k4 = system(
                t_solve[time, location],
                y_solve[time - 1, location] + dt * k3,
                waned,
                p_solve[2 * time, location],
            )

            y_solve[time, location] = y_solve[time - 1, location] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # TODO: Spatial spread goes here!
    return y_solve


@numba.njit
def _compute_waned_this_step(y_past: np.ndarray,
                             vaccine_dist: np.ndarray,
                             natural_dist: np.ndarray,
                             system_size: int) -> np.ndarray:
    waned = np.zeros(2)
    new_vax_immune_index = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, VARIANT_GROUP.total]
    new_r_index = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewR, VARIANT_GROUP.total]
    for i, (index, dist) in enumerate(((new_vax_immune_index, vaccine_dist), (new_r_index, natural_dist))):
        cumulative_total = y_past[:, index] + y_past[:, system_size + index]
        daily_total = cumulative_total[1:] - cumulative_total[:-1]
        waned[i] = (daily_total[::-1] * dist[:-1]).sum()
    return waned


@numba.njit
def _uninterpolate(y_solve: np.ndarray,
                   t_solve: np.ndarray,
                   t: np.ndarray):
    num_dates, num_locs = t.shape
    num_compartments = y_solve.shape[2]
    y_final = np.zeros((num_dates, num_locs, num_compartments))
    for location in np.arange(num_locs):
        for compartment in np.arange(num_compartments):
            y_final[:, location, compartment] = np.interp(
                t[:, location],
                t_solve[:, location],
                y_solve[:, location, compartment]
            )
    return y_final.reshape((num_dates * num_locs, num_compartments))

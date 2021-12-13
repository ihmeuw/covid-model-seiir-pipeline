import functools
import itertools
from typing import List

from loguru import logger
import numba
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    parallel,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    escape_variant,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    TOMBSTONE,
    SYSTEM_TYPE,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
    CHI,
    CHI_NAMES,
    EPI_MEASURE,
    RISK_GROUP,
    VARIANT,
    VARIANT_GROUP,
    TRACKING_COMPARTMENT,
    AGG_INDEX_TYPE,
    TRACKING_COMPARTMENTS,
)
from covid_model_seiir_pipeline.lib.ode_mk2.system import (
    system,
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
)
from covid_model_seiir_pipeline.lib.ode_mk2.utils import (
    cartesian_product,
)


SOLVER_DT: float = 0.1


def run_ode_model(initial_condition: pd.DataFrame,
                  base_parameters: pd.DataFrame,
                  age_scalars: pd.DataFrame,
                  vaccinations: pd.DataFrame,
                  etas: pd.DataFrame,
                  phis: pd.DataFrame,
                  location_ids: List[int],
                  system_type: int,
                  dt: float = SOLVER_DT, 
                  num_cores: int = 5,
                  progress_bar: bool = True):
    # Ensure data frame column labeling is consistent with expected index ordering.
    initial_condition = _sort_columns(initial_condition)
    ics_and_params = [(location_id,
                       initial_condition.loc[location_id],
                       base_parameters.loc[location_id],
                       age_scalars.loc[location_id],
                       vaccinations.loc[location_id],
                       etas.loc[location_id],
                       phis) for location_id in location_ids]

    _runner = functools.partial(
        _run_loc_ode_model,
        dt=dt,
        system_type=system_type,
    )
    results = parallel.run_parallel(
        _runner,
        arg_list=ics_and_params,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    compartments, chis = zip(*results)
    compartments = pd.concat(compartments).reset_index().set_index(['location_id', 'date']).sort_index()
    chis = pd.concat(chis).reset_index().set_index(['location_id', 'date']).sort_index()

    ran = compartments.reset_index().location_id.unique()
    missing = set(location_ids).difference(ran)
    if missing:
        logger.warning(f"Couldn't run locations {missing}")

    return compartments, chis


def _run_loc_ode_model(ic_and_params,
                       dt: float,
                       system_type: int):
    location_id, initial_condition, parameters, age_scalars, vaccines, etas, phis = ic_and_params

    new_e_dates = initial_condition[initial_condition.filter(like='Infection').sum(axis=1) > 0].reset_index().date
    invasion_date = new_e_dates.min()
    ode_start_date = new_e_dates.max()

    t0 = (ode_start_date - invasion_date).days
    if system_type == SYSTEM_TYPE.beta_and_rates:  # This is the forecast
        assert t0 > 0
    else:
        assert t0 == 0

    ode_end_date = initial_condition.loc[initial_condition.iloc[:, 0].notnull()].reset_index().date.max()

    tf = (ode_end_date - invasion_date).days

    dates = initial_condition.reset_index().date

    t = (dates - invasion_date).dt.days.values
    t_solve = np.round(np.arange(t[0], t[-1] + dt, dt), 2)
    t_params = np.round(np.arange(t[0], t[-1] + dt, dt / 2), 2)

    y_solve = _interpolate_y(t0, t, t_solve, initial_condition.to_numpy(), system_type)

    parameters = _interpolate(t, t_params, parameters.to_numpy())
    parameters[np.isnan(parameters)] = TOMBSTONE
    age_scalars = _interpolate(t, t_params, age_scalars.to_numpy())
    vaccines = _interpolate(t, t_params, vaccines.to_numpy())
    etas = _interpolate(t, t_params, etas.to_numpy())
    phis = phis.to_numpy()

    y_solve, chis = _rk45_dde(
        t0, tf,
        t_solve,
        y_solve,
        parameters,
        age_scalars,
        vaccines,
        etas,
        phis,
        system_type,
        dt,
    )
    loc_compartments = pd.DataFrame(_uninterpolate(y_solve, t_solve, t),
                                    columns=initial_condition.columns,
                                    index=initial_condition.index)
    loc_compartments['location_id'] = location_id
    loc_chis = pd.DataFrame(_uninterpolate(chis, t_solve, t),
                            columns=[f'{n}_{r}' for r, n in itertools.product(RISK_GROUP_NAMES, CHI_NAMES)],
                            index=initial_condition.index)
    loc_chis['location_id'] = location_id

    return loc_compartments, loc_chis


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
                   system_type: bool) -> np.ndarray:
    y_solve = np.empty((t_solve.size, y.shape[1]))
    for compartment in np.arange(y.shape[1]):
        if system_type == SYSTEM_TYPE.beta_and_rates:
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
def _rk45_dde(t0: float, tf: float,
              t_solve: np.ndarray,
              y_solve: np.ndarray,
              parameters: np.ndarray,
              age_scalars: np.ndarray,
              vaccines: np.ndarray,
              etas: np.ndarray,
              phis: np.ndarray,
              system_type: int,
              dt: float):
    num_time_points = t_solve.size
    chis = np.zeros((num_time_points, 2 * phis.shape[1]))

    for time in np.arange(num_time_points):
        if not (t0 < t_solve[time] <= tf):
            continue

        chis[time - 1] = compute_chis(time-1, t_solve, y_solve, phis, chis)

        k1 = system(
            t_solve[time - 1],
            y_solve[time - 1],
            parameters[2 * time - 2],
            age_scalars[2 * time - 2],
            vaccines[2 * time - 2],
            etas[2 * time - 2],
            chis[time - 1],
            system_type,
        )

        k2 = system(
            t_solve[time - 1] + dt / 2,
            y_solve[time - 1] + dt / 2 * k1,
            parameters[2 * time - 1],
            age_scalars[2 * time - 2],
            vaccines[2 * time - 1],
            etas[2 * time - 1],
            chis[time - 1],
            system_type,
        )

        k3 = system(
            t_solve[time - 1] + dt / 2,
            y_solve[time - 1] + dt / 2 * k2,
            parameters[2 * time - 1],
            age_scalars[2 * time - 1],
            vaccines[2 * time - 1],
            etas[2 * time - 1],
            chis[time - 1],
            system_type,
        )

        k4 = system(
            t_solve[time],
            y_solve[time - 1] + dt * k3,
            parameters[2 * time],
            age_scalars[2 * time],
            vaccines[2 * time],
            etas[2 * time],
            chis[time - 1],
            system_type,
        )

        y_solve[time] = escape_variant.maybe_invade(
            t_solve[time],
            y_solve[time - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4),
            parameters[2 * time],
        )

    return y_solve, chis


@numba.njit
def _uninterpolate(y_solve: np.ndarray,
                   t_solve: np.ndarray,
                   t: np.ndarray):
    y_final = np.zeros((len(t), y_solve.shape[1]))
    for compartment in np.arange(y_solve.shape[1]):
        y_final[:, compartment] = np.interp(t, t_solve, y_solve[:, compartment])
    return y_final


@numba.njit
def compute_chis(time, t_solve, y_solve, phis, chis):
    if t_solve[time] % 1:
        return chis[time - 1]

    num_chis = CHI.max() + 1
    chi = np.zeros(len(RISK_GROUP) * num_chis)
    for risk_group in RISK_GROUP:
        group_size = y_solve.shape[1] // len(RISK_GROUP)
        group_start = risk_group * group_size
        group_end = (risk_group + 1) * group_size
        group_y = y_solve[:time:10, group_start:group_end]
        group_chi = np.zeros(num_chis)
        t_end = min(group_y.shape[0], phis.shape[0]) - 1

        for from_variant in VARIANT[1:]:
            idx = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Infection, VARIANT_GROUP.all,
                                        from_variant, AGG_INDEX_TYPE.all]
            cumulative_new_e_variant = group_y[:, idx]
            denominator = cumulative_new_e_variant[-1]

            if denominator:                
                for epi_measure, to_variant in cartesian_product((np.array(EPI_MEASURE), np.array(VARIANT[1:]))):
                    numerator = 0.
                    idx = CHI[from_variant, to_variant, epi_measure]

                    for tau in range(1, t_end):
                        numerator += (
                            (cumulative_new_e_variant[-tau] - cumulative_new_e_variant[-tau - 1]) * phis[tau, idx]
                        )

                    group_chi[idx] = numerator / denominator

        group_chi_start = risk_group * num_chis
        group_chi_end = (risk_group + 1) * num_chis

        chi[group_chi_start:group_chi_end] = group_chi

    if DEBUG:
        assert np.all(np.isfinite(chi))
        assert np.all(chi >= 0.)
        assert np.all(chi <= 1.)

    return chi

import numba
import numpy as np

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    RISK_GROUP,
    BASE_PARAMETER,
    VARIANT,
    VARIANT_GROUP,
    VACCINE_STATUS,
    COMPARTMENT,
    # Indexing arrays
    PARAMETERS,
    NEW_E,
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,

    # Debug flag
    DEBUG,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    accounting,
    escape_variant,
    parameters,
)


@numba.njit
def fit_system(t: float, y: np.ndarray, waned: np.ndarray, input_parameters: np.ndarray):
    return _system(t, y, waned, input_parameters, forecast=False)


@numba.njit
def forecast_system(t: float, y: np.ndarray, waned: np.ndarray, input_parameters: np.ndarray):
    return _system(t, y, waned, input_parameters, forecast=True)


@numba.njit
def _system(t: float,
            y: np.ndarray,
            input_parameters: np.ndarray,
            phis: np.ndarray,
            forecast: bool):
    aggregates = parameters.make_aggregates(y)
    params = input_parameters[:PARAMETERS.max()+1]
    vaccines = input_parameters[PARAMETERS.max()+1:]

    if DEBUG:
        assert np.all(np.isfinite(aggregates))
        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(vaccines))

    dy = np.zeros_like(y)
    
    system_size = TRACKING_COMPARTMENTS.max() + 1
    for i in range(len(RISK_GROUP)):
        group_start = i * system_size
        group_end = (i + 1) * system_size
        group_vaccine_start = i * 2
        group_vaccine_end = (i + 1) * 2

        group_y = y[group_start:group_end]
        group_vaccines = vaccines[group_vaccine_start:group_vaccine_end]

        new_e, effective_susceptible = parameters.make_new_e(
            t,
            group_y,
            params,
            aggregates,
            phis,
            forecast,
        )

        group_dy = _single_group_system(
            t,
            group_y,
            new_e,
            effective_susceptible,
            params,
            group_vaccines,
        )

        group_dy = escape_variant.maybe_invade(
            group_y,
            group_dy,
            aggregates,
            params,
        )

        dy[group_start:group_end] = group_dy

    if DEBUG:
        assert np.all(np.isfinite(dy))

    return dy


@numba.njit
def _single_group_system(t: float,
                         group_y: np.ndarray,
                         new_e: np.ndarray,
                         effective_susceptible: np.ndarray,
                         params: np.ndarray,
                         group_vaccines: np.ndarray):
    transition_map = np.zeros((group_y.size, group_y.size))

    sigma = params[PARAMETERS[BASE_PARAMETER.sigma, VARIANT_GROUP.all]]
    gamma = params[PARAMETERS[BASE_PARAMETER.sigma, VARIANT_GROUP.all]]

    vaccine_eligible = np.zeros(len(VACCINE_STATUS))
    # Transmission
    for vaccine_status in VACCINE_STATUS:
        for variant_to in VARIANT:
            e_idx = COMPARTMENTS[COMPARTMENT.E, variant_to, vaccine_status]
            i_idx = COMPARTMENTS[COMPARTMENT.I, variant_to, vaccine_status]
            s_to_idx = COMPARTMENTS[COMPARTMENT.S, variant_to, vaccine_status]

            for variant_from in VARIANT:
                s_from_idx = COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]
                transition_map[s_from_idx, e_idx] += new_e[NEW_E[variant_from, variant_to, vaccine_status]]
            transition_map[e_idx, i_idx] += sigma * group_y[e_idx]
            transition_map[i_idx, s_to_idx] += gamma * group_y[i_idx]

            vaccine_eligible[vaccine_status] += group_y[s_to_idx]

    for vaccine_status in VACCINE_STATUS[:-1]:
        for variant in VARIANT:
            new_e_from_s = new_e[NEW_E[variant].flatten()].sum()
            s_from_idx = COMPARTMENTS[COMPARTMENT.S, variant, vaccine_status]
            s_to_idx = COMPARTMENTS[COMPARTMENT.S, variant, vaccine_status + 1]
            expected_vaccines = (
                math.safe_divide(group_y[s_from_idx], vaccine_eligible[vaccine_status])
                * group_vaccines[vaccine_status]
            )
            to_vaccinate = min(expected_vaccines, group_y[s_from_idx] - new_e_from_s)
            transition_map[s_from_idx, s_to_idx] += to_vaccinate

    inflow = transition_map.sum(axis=0)
    outflow = transition_map.sum(axis=1)
    group_dy = inflow - outflow

    if DEBUG:
        assert np.all(np.isfinite(group_dy))
        assert np.all(group_y + group_dy >= -1e-7)
        assert group_dy.sum() < 1e-5
       
    group_dy = accounting.compute_tracking_compartments(
        t, 
        group_dy,
        new_e,
        effective_susceptible,
        transition_map,
    )

    return group_dy

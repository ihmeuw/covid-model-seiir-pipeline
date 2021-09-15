import numba
import numpy as np

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    RISK_GROUP,
    BASE_COMPARTMENT,
    BASE_PARAMETER,
    VARIANT,
    VARIANT_GROUP,
    SUSCEPTIBLE_TYPE,
    PROTECTION_STATUS,
    IMMUNE_STATUS,
    VACCINE_TYPE,
    VACCINATION_STATUS,
    REMOVED_VACCINATION_STATUS,
    AGG_WANED,
    # Indexing arrays
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,
    PARAMETERS,
    AGGREGATES,
    # Debug flag
    DEBUG,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    accounting,
    escape_variant,
    parameters,
    vaccinations,
)


@numba.njit
def fit_system(t: float, y: np.ndarray, waned: np.ndarray, input_parameters: np.ndarray):
    return _system(t, y, waned, input_parameters, forecast=False)


@numba.njit
def forecast_system(t: float, y: np.ndarray, waned: np.ndarray, input_parameters: np.ndarray):
    return _system(t, y, waned, input_parameters, forecast=True)


@numba.njit
def _system(t: float, y: np.ndarray, waned: np.ndarray, input_parameters: np.ndarray, forecast: bool):
    aggregates = parameters.make_aggregates(y)
    params, vaccines, force_of_infection = parameters.normalize_parameters(
        input_parameters,
        aggregates,
        forecast,
    )

    dy = np.zeros_like(y)
    
    system_size = TRACKING_COMPARTMENTS.max() + 1
    for i in range(len(RISK_GROUP)):
        group_start = i * system_size
        group_end = (i + 1) * system_size
        group_vaccine_start = i * len(VACCINE_TYPE)
        group_vaccine_end = (i + 1) * len(VACCINE_TYPE)

        group_y = y[group_start:group_end]
        group_vaccines = vaccines[group_vaccine_start:group_vaccine_end]

        group_dy = _single_group_system(
            t,
            group_y,
            force_of_infection,
            waned,
            aggregates,
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
                         force_of_infection: np.ndarray,
                         waned: np.ndarray,
                         aggregates: np.ndarray,
                         params: np.ndarray,
                         group_vaccines: np.ndarray):
    transition_map = np.zeros((group_y.size, group_y.size))
    vaccines_out = vaccinations.allocate(
        group_y,
        group_vaccines,
        force_of_infection,
    )

    transition_map = do_vaccination(
        vaccines_out,
        transition_map,
    )

    sigma = params[PARAMETERS[BASE_PARAMETER.sigma, VARIANT_GROUP.all]]
    gamma = params[PARAMETERS[BASE_PARAMETER.gamma, VARIANT_GROUP.all]]
    for variant in VARIANT:
        transition_map = do_transmission(
            variant,
            group_y,
            sigma,
            gamma,
            force_of_infection[variant],
            transition_map,
        )

        transition_map = do_natural_immunity_waning(
            variant,
            group_y,
            waned[0],
            aggregates[AGGREGATES[BASE_COMPARTMENT.R, variant]],
            transition_map,
        )

    transition_map = do_vaccine_immunity_waning(
        group_y,
        waned[1],
        waned[1],
        transition_map,
    )

    inflow = transition_map.sum(axis=0)
    outflow = transition_map.sum(axis=1)
    group_dy = inflow - outflow

    if DEBUG:
        assert np.all(np.isfinite(group_dy))
        assert np.all(group_y + group_dy >= -1e-10)
        assert group_dy.sum() < 1e-5

    group_dy = accounting.compute_tracking_compartments(
        group_dy,
        transition_map,
    )

    return group_dy


@numba.njit
def do_vaccination(
    vaccines_out: np.ndarray,
    transition_map: np.ndarray,
) -> np.ndarray:
    for vaccine_type in VACCINE_TYPE:
        for current_status in PROTECTION_STATUS:
            from_compartment = COMPARTMENTS[BASE_COMPARTMENT.S, current_status, VACCINATION_STATUS.unvaccinated]
            to_compartment = COMPARTMENTS[BASE_COMPARTMENT.S, vaccine_type, VACCINATION_STATUS.vaccinated]
            transition_map[from_compartment, to_compartment] += vaccines_out[from_compartment, vaccine_type]
        for variant in VARIANT:
            from_compartment = COMPARTMENTS[BASE_COMPARTMENT.R, variant, REMOVED_VACCINATION_STATUS.unvaccinated]
            if vaccine_type == VACCINE_TYPE.unprotected:
                to_compartment = COMPARTMENTS[BASE_COMPARTMENT.R, variant, REMOVED_VACCINATION_STATUS.newly_vaccinated]
            else:
                to_compartment = COMPARTMENTS[BASE_COMPARTMENT, vaccine_type, VACCINATION_STATUS.vaccinated]
            transition_map[from_compartment, to_compartment] += vaccines_out[from_compartment, vaccine_type]
    return transition_map


@numba.njit
def do_transmission(
    variant: str,
    group_y: np.ndarray,
    sigma: float,
    gamma: float,
    force_of_infection: float,
    transition_map: np.ndarray,
) -> np.ndarray:    
    for vaccination_status in VACCINATION_STATUS:
        e_idx = COMPARTMENTS[BASE_COMPARTMENT.E, variant, vaccination_status]
        i_idx = COMPARTMENTS[BASE_COMPARTMENT.I, variant, vaccination_status]
        r_idx = COMPARTMENTS[BASE_COMPARTMENT.R, variant, vaccination_status]

        for susceptible_type in SUSCEPTIBLE_TYPE:
            s_idx = COMPARTMENTS[BASE_COMPARTMENT.S, susceptible_type, vaccination_status]
            transition_map[s_idx, e_idx] = group_y[s_idx] * force_of_infection

        transition_map[e_idx, i_idx] = sigma * group_y[e_idx]
        transition_map[i_idx, r_idx] = gamma * group_y[i_idx]
    return transition_map


@numba.njit
def do_natural_immunity_waning(
    variant: str,
    group_y: np.ndarray,
    natural_immunity_waned: float,
    r_variant: float,
    transition_map: np.ndarray,
) -> np.ndarray:

    for vaccination_status in REMOVED_VACCINATION_STATUS:
        from_index = COMPARTMENTS[BASE_COMPARTMENT.R, variant, vaccination_status]
        waned = math.safe_divide(group_y[from_index], r_variant) * natural_immunity_waned

        # TODO: make a parameter
        protection_fraction = 1 / len(PROTECTION_STATUS)

        for protection_status in PROTECTION_STATUS:
            s_vaccination_status = min(VACCINATION_STATUS.vaccinated, vaccination_status)
            s_idx = COMPARTMENTS[BASE_COMPARTMENT.S, protection_status, s_vaccination_status]
            transition_map[from_index, s_idx] = protection_fraction * waned

    return transition_map


@numba.njit
def do_vaccine_immunity_waning(
    group_y: np.ndarray,
    vaccine_immunity_waned: float,
    vaccine_immune: float,
    transition_map: np.ndarray,
) -> np.ndarray:
    for immunity_status in IMMUNE_STATUS:
        from_index = COMPARTMENTS[BASE_COMPARTMENT.S, immunity_status, VACCINATION_STATUS.vaccinated]
        waned = math.safe_divide(group_y[from_index], vaccine_immune) * vaccine_immunity_waned

        # TODO: make a parameter
        protection_fraction = 1 / len(PROTECTION_STATUS)

        for protection_status in PROTECTION_STATUS:
            to_index = COMPARTMENTS[BASE_COMPARTMENT.S, protection_status, VACCINATION_STATUS.vaccinated]
            transition_map[from_index, to_index] = protection_fraction * waned
    return transition_map

import numba
import numpy as np


from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    RISK_GROUP,
    BASE_PARAMETER,
    VARIANT,
    VARIANT_GROUP,
    VACCINE_STATUS,
    COMPARTMENT,
    EPI_MEASURE,
    # Indexing arrays
    PARAMETERS,
    NEW_E,
    COMPARTMENTS,
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
    Printer
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    accounting,
    escape_variant,
    parameters,
    utils,
)


@numba.njit
def system(t: float,
           y: np.ndarray,           
           params: np.ndarray,
           rates: np.ndarray,
           vaccines: np.ndarray,
           etas: np.ndarray,
           chis: np.ndarray,
           forecast: bool):
    aggregates = parameters.make_aggregates(y)
    new_e, effective_susceptible, beta, outcomes = parameters.make_new_e(
        t,
        y,
        params,
        rates,
        aggregates,
        etas,
        chis,
        forecast,
    )  

    if DEBUG:
        assert np.all(np.isfinite(aggregates))
        # assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(vaccines))
        assert np.all(np.isfinite(etas))
        assert np.all(np.isfinite(chis))

    dy = np.zeros_like(y)
    
    system_size = y.size // len(RISK_GROUP)
    for risk_group in range(len(RISK_GROUP)):
        group_start = risk_group * system_size
        group_end = (risk_group + 1) * system_size
        
        group_y = utils.subset_risk_group(y, risk_group)        
        group_vaccines = utils.subset_risk_group(vaccines, risk_group)
        group_new_e = utils.subset_risk_group(new_e, risk_group)
        group_effective_susceptible = utils.subset_risk_group(effective_susceptible, risk_group)
        group_outcomes = utils.subset_risk_group(outcomes, risk_group)

        group_dy = _single_group_system(
            t,
            group_y,
            group_new_e,
            group_effective_susceptible,
            beta,
            params,
            group_vaccines,
            group_outcomes,
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
        assert np.any(dy > 0.)

    return dy


@numba.njit
def _single_group_system(t: float,
                         group_y: np.ndarray,
                         new_e: np.ndarray,
                         effective_susceptible: np.ndarray,
                         beta: float,
                         params: np.ndarray,
                         group_vaccines: np.ndarray,
                         group_outcomes: np.ndarray,):
    transition_map = np.zeros((group_y.size, group_y.size))

    sigma = params[PARAMETERS[BASE_PARAMETER.sigma, VARIANT_GROUP.all, EPI_MEASURE.infection]]
    gamma = params[PARAMETERS[BASE_PARAMETER.sigma, VARIANT_GROUP.all, EPI_MEASURE.infection]]

    vaccine_eligible = np.zeros(len(VACCINE_STATUS))
    # Transmission
    for vaccine_status in VACCINE_STATUS:
        for variant_to in VARIANT:
            e_idx = COMPARTMENTS[COMPARTMENT.E, variant_to, vaccine_status]
            i_idx = COMPARTMENTS[COMPARTMENT.I, variant_to, vaccine_status]
            s_to_idx = COMPARTMENTS[COMPARTMENT.S, variant_to, vaccine_status]

            for variant_from in VARIANT:
                s_from_idx = COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]
                transition_map[s_from_idx, e_idx] += new_e[NEW_E[vaccine_status, variant_from, variant_to]]
            transition_map[e_idx, i_idx] += sigma * group_y[e_idx]
            transition_map[i_idx, s_to_idx] += gamma * group_y[i_idx]

            vaccine_eligible[vaccine_status] += group_y[s_to_idx]

    for vaccine_status in VACCINE_STATUS[:-1]:
        for variant in VARIANT:
            new_e_from_s = new_e[NEW_E[vaccine_status, variant].flatten()].sum()
            s_from_idx = COMPARTMENTS[COMPARTMENT.S, variant, vaccine_status]
            s_to_idx = COMPARTMENTS[COMPARTMENT.S, variant, vaccine_status + 1]
            expected_vaccines = (
                utils.safe_divide(group_y[s_from_idx], vaccine_eligible[vaccine_status])
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
        beta,
        group_outcomes,
        transition_map,
    )

    return group_dy




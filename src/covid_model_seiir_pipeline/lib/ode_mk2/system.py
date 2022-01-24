import numba
import numpy as np


from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    TOMBSTONE,
    # Indexing tuples
    RISK_GROUP,
    EPI_VARIANT_PARAMETER,
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
    Printer,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    accounting,
    parameters,
    utils,
)


@numba.njit
def system(t: float,
           y: np.ndarray,           
           params: np.ndarray,
           age_scalars: np.ndarray,
           vaccines: np.ndarray,
           etas: np.ndarray,
           chis: np.ndarray,
           system_type: int):

    if DEBUG:
        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(vaccines))
        assert np.all(vaccines >= 0.)
        assert np.all(np.isfinite(etas))
        assert np.all(etas >= 0.)
        assert np.all(etas <= 1.)
        assert np.all(np.isfinite(chis))
        assert np.all(chis >= 0.)
        assert np.all(chis <= 1.)

    aggregates = parameters.make_aggregates(y)
    new_e, effective_susceptible, beta, rates = parameters.compute_intermediate_epi_parameters(
        t,
        y,
        params,
        age_scalars,
        aggregates,
        etas,
        chis,
        system_type,
    )  

    dy = np.zeros_like(y)
    
    system_size = y.size // len(RISK_GROUP)
    for risk_group in range(len(RISK_GROUP)):
        group_start = risk_group * system_size
        group_end = (risk_group + 1) * system_size
        
        group_y = utils.subset_risk_group(y, risk_group)        
        group_vaccines = utils.subset_risk_group(vaccines, risk_group)
        group_new_e = utils.subset_risk_group(new_e, risk_group)
        group_effective_susceptible = utils.subset_risk_group(effective_susceptible, risk_group)
        group_rates = utils.subset_risk_group(rates, risk_group)

        transition_map = single_group_system(
            t,
            group_y,
            group_new_e,
            params,
            group_vaccines,
        )

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
            group_new_e,
            group_effective_susceptible,
            beta,
            group_rates,
            transition_map,
        )

        dy[group_start:group_end] = group_dy

    if DEBUG:
        assert np.any(dy > 0.)

    return dy


@numba.njit
def single_group_system(t: float,
                        group_y: np.ndarray,
                        new_e: np.ndarray,
                        params: np.ndarray,
                        group_vaccines: np.ndarray):
    transition_map = np.zeros((group_y.size, group_y.size))

    vaccine_eligible = np.zeros(len(VACCINE_STATUS))
    # Transmission
    for variant_to, vaccine_status in utils.cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS))):
        e_idx = COMPARTMENTS[COMPARTMENT.E, vaccine_status, variant_to]
        i_idx = COMPARTMENTS[COMPARTMENT.I, vaccine_status, variant_to]
        s_to_idx = COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant_to]
        
        sigma = params[PARAMETERS[EPI_VARIANT_PARAMETER.sigma, variant_to, EPI_MEASURE.infection]]
        gamma = params[PARAMETERS[EPI_VARIANT_PARAMETER.gamma, variant_to, EPI_MEASURE.infection]]

        for variant_from in VARIANT:
            s_from_idx = COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant_from]
            transition_map[s_from_idx, e_idx] += new_e[NEW_E[vaccine_status, variant_from, variant_to]]

        transition_map[e_idx, i_idx] += sigma * group_y[e_idx]
        transition_map[i_idx, s_to_idx] += gamma * group_y[i_idx]
        vaccine_eligible[vaccine_status] += group_y[s_to_idx]

    for variant_from, vaccine_status in utils.cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS[:-1]))):
        new_e_from_s = 0.
        for variant_to in VARIANT:
            new_e_from_s += new_e[NEW_E[vaccine_status, variant_from, variant_to]]
        s_from_idx = COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant_from]
        s_to_idx = COMPARTMENTS[COMPARTMENT.S, vaccine_status + 1, variant_from]
        expected_vaccines = (
            utils.safe_divide(group_y[s_from_idx], vaccine_eligible[vaccine_status])
            * group_vaccines[vaccine_status]
        )
        to_vaccinate = max(min(expected_vaccines, group_y[s_from_idx] - new_e_from_s), 0.)
        transition_map[s_from_idx, s_to_idx] += to_vaccinate

    return transition_map

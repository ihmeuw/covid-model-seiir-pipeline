import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT,
    IMMUNE_STATUS,
    VACCINATION_STATUS,
    REMOVED_VACCINATION_STATUS,
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,
    COMPARTMENT_GROUPS,
    DEBUG,
)


@numba.njit
def compute_tracking_compartments(group_dy: np.ndarray, transition_map: np.ndarray) -> np.ndarray:
    for variant in VARIANT:
        ##################
        # New infections #
        ##################
        new_e_unvaccinated = transition_map[:, COMPARTMENTS[('E', variant, 'unvaccinated')]].sum()
        new_e_vaccinated = transition_map[:, COMPARTMENTS[('E', variant, 'vaccinated')]].sum()

        group_dy[TRACKING_COMPARTMENTS[('NewE', 'unvaccinated')]] += new_e_unvaccinated
        group_dy[TRACKING_COMPARTMENTS[('NewE', 'vaccinated')]] += new_e_vaccinated
        group_dy[TRACKING_COMPARTMENTS[('NewE', variant)]] += new_e_unvaccinated + new_e_vaccinated
        group_dy[TRACKING_COMPARTMENTS[('NewE', 'total')]] += new_e_unvaccinated + new_e_vaccinated

        #########################
        # New recovered/removed #
        #########################
        for vaccination_status in VACCINATION_STATUS:
            new_r = transition_map[:, COMPARTMENTS[('R', variant, vaccination_status)]].sum()
            group_dy[TRACKING_COMPARTMENTS[('NewR', variant)]] += new_r
            group_dy[TRACKING_COMPARTMENTS[('NewR', 'total')]] += new_r

        ##############################
        # New natural immunity waned #
        ##############################
        for vaccination_status in REMOVED_VACCINATION_STATUS:
            waned = transition_map[COMPARTMENTS[('R', variant, vaccination_status)],
                                   COMPARTMENT_GROUPS[('S', 'total')]].sum(axis=1)
            group_dy[TRACKING_COMPARTMENTS[('Waned', variant)]] += waned
            group_dy[TRACKING_COMPARTMENTS[('Waned', 'natural')]] += waned

    ######################
    # New vaccine immune #
    ######################
    for immune_status in IMMUNE_STATUS:
        group_dy[TRACKING_COMPARTMENTS[('NewVaxImmune', immune_status)]] += (
            transition_map[:, COMPARTMENTS[('S', immune_status, 'vaccinated')]].sum()
        )

        ##############################
        # New vaccine immunity waned #
        ##############################
        waned = transition_map[COMPARTMENTS[('S', immune_status, 'vaccinated')],
                               COMPARTMENT_GROUPS[('S', 'non_immune')]].sum(axis=1)
        group_dy[TRACKING_COMPARTMENTS[('Waned', immune_status)]] += waned
        group_dy[TRACKING_COMPARTMENTS[('Waned', 'vaccine')]] += waned

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

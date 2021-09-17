import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    BASE_COMPARTMENT,
    TRACKING_COMPARTMENT,
    VARIANT,
    VARIANT_GROUP,
    IMMUNE_STATUS,
    VACCINE_TYPE,
    VACCINATION_STATUS,
    REMOVED_VACCINATION_STATUS,
    AGG_IMMUNE_STATUS,
    AGG_VACCINATION_STATUS,
    AGG_WANED,
    # Indexing arrays
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,
    # Compartment groups
    CG_SUSCEPTIBLE,
    # Debug flag
    DEBUG,
)


@numba.njit
def compute_tracking_compartments(t: float, group_dy: np.ndarray, transition_map: np.ndarray) -> np.ndarray:
    if t > 500:
        import pdb; pdb.set_trace()
    for variant in VARIANT:
        for vaccination_status, agg_vaccination_status in zip(VACCINATION_STATUS, AGG_VACCINATION_STATUS):
            ##################
            # New Infections #
            ##################
            new_e = transition_map[:, COMPARTMENTS[BASE_COMPARTMENT.E, variant, vaccination_status]].sum()
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, agg_vaccination_status]] += new_e
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, variant]] += new_e
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, VARIANT_GROUP.total]] += new_e
            #########################
            # New Recovered/Removed #
            #########################
            new_r = transition_map[:, COMPARTMENTS[BASE_COMPARTMENT.R, variant, vaccination_status]].sum()
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewR, variant]] += new_r
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewR, VARIANT_GROUP.total]] += new_r

        for vaccination_status in REMOVED_VACCINATION_STATUS:
            #############
            # New waned #
            #############
            waned = transition_map[COMPARTMENTS[BASE_COMPARTMENT.R, variant, vaccination_status],
                                   CG_SUSCEPTIBLE[AGG_IMMUNE_STATUS.non_immune]].sum()
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, variant]] += waned
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, AGG_WANED.natural]] += waned

    ######################
    # New vaccine immune #
    ######################
    for immune_status, agg_immune_status in zip(IMMUNE_STATUS, AGG_IMMUNE_STATUS):
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaxImmune, agg_immune_status]] += (
            transition_map[:, COMPARTMENTS[BASE_COMPARTMENT.S, immune_status, VACCINATION_STATUS.vaccinated]].sum()
        )

        ##############################
        # New vaccine immunity waned #
        ##############################
        waned = transition_map[COMPARTMENTS[BASE_COMPARTMENT.S, immune_status, VACCINATION_STATUS.vaccinated],
                               CG_SUSCEPTIBLE[AGG_IMMUNE_STATUS.non_immune]].sum()
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, agg_immune_status]] += waned
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, AGG_WANED.vaccine]] += waned

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

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
    AGG_VACCINE_TYPE,
    AGG_VACCINATION_STATUS,
    AGG_WANED,
    AGG_OTHER,
    # Indexing arrays
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,
    # Compartment groups
    CG_SUSCEPTIBLE,
    CG_TOTAL,
    # Debug flag
    DEBUG,
)


@numba.njit
def compute_tracking_compartments(t: float, group_dy: np.ndarray, transition_map: np.ndarray) -> np.ndarray:
#     if t > 500:
#         import pdb; pdb.set_trace()
    for variant in VARIANT:
        for vaccination_status, agg_vaccination_status in zip(VACCINATION_STATUS, AGG_VACCINATION_STATUS):
            ##################
            # New Infections #
            ##################
            new_e = transition_map[CG_SUSCEPTIBLE[AGG_OTHER.total], 
                                   COMPARTMENTS[BASE_COMPARTMENT.E, variant, vaccination_status]].sum()
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
                                   CG_SUSCEPTIBLE[AGG_OTHER.non_immune]].sum()
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, variant]] += waned
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, AGG_WANED.natural]] += waned

    ######################
    # New vaccine immune #
    ######################
    for vaccine_type, agg_vaccine_type in zip(VACCINE_TYPE, AGG_VACCINE_TYPE):
        new_vaccines = transition_map[CG_TOTAL[AGG_OTHER.vaccine_eligible], 
                                      COMPARTMENTS[BASE_COMPARTMENT.S, vaccine_type, VACCINATION_STATUS.vaccinated]].sum()
        if vaccine_type == VACCINE_TYPE.unprotected:
            new_vaccines += transition_map[CG_TOTAL[AGG_OTHER.vaccine_eligible]][:, 
                                           COMPARTMENTS[BASE_COMPARTMENT.R, VARIANT, REMOVED_VACCINATION_STATUS.newly_vaccinated]].sum()
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, agg_vaccine_type]] += new_vaccines
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, AGG_OTHER.total]] += new_vaccines

    for immune_status, agg_immune_status in zip(IMMUNE_STATUS, AGG_IMMUNE_STATUS):
        ##############################
        # New vaccine immunity waned #
        ##############################
        waned = transition_map[COMPARTMENTS[BASE_COMPARTMENT.S, immune_status, VACCINATION_STATUS.vaccinated],
                               CG_SUSCEPTIBLE[AGG_OTHER.non_immune]].sum()        
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, agg_immune_status]] += waned
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.Waned, AGG_WANED.vaccine]] += waned

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    COMPARTMENT,
    TRACKING_COMPARTMENT,
    VARIANT,
    VACCINE_STATUS,
    # Indexing arrays
    NEW_E,
    EFFECTIVE_SUSCEPTIBLE,
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,    
    # Debug flag
    DEBUG,
)


@numba.njit
def compute_tracking_compartments(t: float,
                                  group_dy: np.ndarray,
                                  new_e: np.ndarray,
                                  effective_susceptible: np.ndarray,
                                  transition_map: np.ndarray) -> np.ndarray:
    for variant in VARIANT:
        for vaccine_status in VACCINE_STATUS:            
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, variant, vaccine_status]] += (
                new_e[NEW_E[vaccine_status, variant].flatten()].sum()
            )
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.EffectiveSusceptible, variant, vaccine_status]] += (
                effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant, vaccine_status]]
            )

        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, variant, VACCINE_STATUS.unvaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.unvaccinated],
                           COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.vaccinated]]
        )
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewBooster, variant, VACCINE_STATUS.vaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.vaccinated],
                           COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.booster]]
        )

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

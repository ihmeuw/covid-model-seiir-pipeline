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
            group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, variant]] += new_e[NEW_E[vaccine_status, variant].flatten()].sum()
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, variant]] += (transition_map[
                COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.unvaccinated],
                COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.vaccinated],
        ])
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewBooster, variant]] += (transition_map[
            COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.vaccinated],
            COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.booster],
        ])
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.EffectiveSusceptible, variant]] += (
            effective_susceptible[variant]
        )

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

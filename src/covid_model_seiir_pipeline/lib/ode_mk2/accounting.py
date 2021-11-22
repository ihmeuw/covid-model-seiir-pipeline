import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    COMPARTMENT,
    TRACKING_COMPARTMENT,
    TRACKING_COMPARTMENT_TYPE,
    VARIANT,
    VARIANT_INDEX_TYPE,
    VACCINE_STATUS,
    VACCINE_INDEX_TYPE,
    EPI_MEASURE,
    REPORTED_EPI_MEASURE,
    AGG_INDEX_TYPE,
    # Indexing arrays
    NEW_E,
    EFFECTIVE_SUSCEPTIBLE,
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
    Printer
)


@numba.njit
def compute_tracking_compartments(t: float,
                                  group_dy: np.ndarray,
                                  new_e: np.ndarray,
                                  effective_susceptible: np.ndarray,
                                  beta: np.ndarray,
                                  group_outcomes: np.ndarray,
                                  transition_map: np.ndarray) -> np.ndarray:
    for variant in VARIANT:
        for vaccine_status in VACCINE_STATUS:
            for variant_from in VARIANT:
                group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, variant, vaccine_status]] += (
                    new_e[NEW_E[vaccine_status, variant_from, variant]]
                )
                group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewE, variant, VACCINE_INDEX_TYPE.all]] += (
                    new_e[NEW_E[vaccine_status, variant_from, variant]]
                )
                group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.EffectiveSusceptible, variant, vaccine_status]] += (
                    effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_from, variant, vaccine_status]]
                )

        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewVaccination, variant, VACCINE_STATUS.unvaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.unvaccinated],
                           COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.vaccinated]]
        )
        group_dy[TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT.NewBooster, variant, VACCINE_STATUS.vaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.vaccinated],
                           COMPARTMENTS[COMPARTMENT.S, variant, VACCINE_STATUS.booster]]
        )    
    for agg_idx, epi_idx in ((AGG_INDEX_TYPE.all, EPI_MEASURE.infection),
                             (AGG_INDEX_TYPE.death, EPI_MEASURE.death),
                             (AGG_INDEX_TYPE.admission, EPI_MEASURE.admission),
                             (AGG_INDEX_TYPE.case, EPI_MEASURE.case)):
        beta_idx = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT_TYPE.beta, VARIANT_INDEX_TYPE.none, agg_idx]        
        group_dy[beta_idx] = beta[epi_idx]

    for compartment, epi_idx in ((TRACKING_COMPARTMENT_TYPE.infection, EPI_MEASURE.infection),
                                 (TRACKING_COMPARTMENT_TYPE.death, EPI_MEASURE.death),
                                 (TRACKING_COMPARTMENT_TYPE.admission, EPI_MEASURE.admission),
                                 (TRACKING_COMPARTMENT_TYPE.case, EPI_MEASURE.case)):
        idx = TRACKING_COMPARTMENTS[compartment, VARIANT.ancestral, VACCINE_INDEX_TYPE.all]
        group_dy[idx] = group_outcomes[epi_idx]

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

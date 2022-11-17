import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    COMPARTMENT,
    TRACKING_COMPARTMENT,
    TRACKING_COMPARTMENT_TYPE,
    VARIANT,
    VARIANT_GROUP,
    VACCINE_STATUS,
    EPI_MEASURE,
    AGG_INDEX_TYPE,
    # Indexing arrays
    NEW_E,
    RATES,
    EFFECTIVE_SUSCEPTIBLE,
    COMPARTMENTS,
    TRACKING_COMPARTMENTS,
    # Mappings
    BETA_MAP,
    VACCINE_STATUS_MAP,
    EPI_MEASURE_MAP,
    VACCINE_COUNT_MAP,
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
    Printer
)
from covid_model_seiir_pipeline.lib.ode_mk2.utils import (
    cartesian_product,
)

@numba.njit
def compute_tracking_compartments(t: float,
                                  dy: np.ndarray,
                                  new_e: np.ndarray,
                                  effective_susceptible: np.ndarray,
                                  beta: np.ndarray,
                                  rates: np.ndarray,
                                  transition_map: np.ndarray) -> np.ndarray:
    for agg_idx, epi_idx in BETA_MAP:
        beta_idx = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT_TYPE.Beta, VARIANT.none, VARIANT.none, agg_idx]
        dy[beta_idx] = beta[epi_idx]

    for tc, epi_measure in EPI_MEASURE_MAP:
        for variant_from, variant_to in cartesian_product((np.array(VARIANT), np.array(VARIANT))):
            for agg_vaccine_status, vaccine_status in VACCINE_STATUS_MAP:
                infections = new_e[NEW_E[vaccine_status, variant_from, variant_to]]
                if epi_measure == EPI_MEASURE.infection:
                    rate = 1.0
                else:
                    rate = rates[RATES[vaccine_status, variant_from, variant_to, epi_measure]]
                measure = infections * rate
                if variant_from == VARIANT.none:
                    dy[TRACKING_COMPARTMENTS[tc, VARIANT.none, VARIANT_GROUP.all, AGG_INDEX_TYPE.all]] += measure
                    if vaccine_status == VACCINE_STATUS.course_0:
                        dy[TRACKING_COMPARTMENTS[tc, variant_from, VARIANT_GROUP.all, agg_vaccine_status]] += measure
                        dy[TRACKING_COMPARTMENTS[tc, variant_from, variant_to, agg_vaccine_status]] += measure
                dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.all]] += measure
                dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, VARIANT_GROUP.all, agg_vaccine_status]] += measure
                dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, variant_to, AGG_INDEX_TYPE.all]] += measure

    tc_effs = TRACKING_COMPARTMENT.EffectiveSusceptible
    for variant in VARIANT:
        for tc, agg_idx, from_vax, to_vax in VACCINE_COUNT_MAP:
            dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, VARIANT_GROUP.all, agg_idx]] += (
                transition_map[COMPARTMENTS[COMPARTMENT.S, from_vax, variant],
                               COMPARTMENTS[COMPARTMENT.S, to_vax, variant]]
            )

        for variant_from, vaccine_status in cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS))):
            dy[TRACKING_COMPARTMENTS[tc_effs, VARIANT_GROUP.all, variant, AGG_INDEX_TYPE.all]] += (
                effective_susceptible[EFFECTIVE_SUSCEPTIBLE[vaccine_status, variant_from, variant]]
            )

    if DEBUG:
        assert np.all(np.isfinite(dy))
    return dy

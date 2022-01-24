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
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
    Printer
)
from covid_model_seiir_pipeline.lib.ode_mk2.utils import (
    cartesian_product,
)

BETA_MAP = ((AGG_INDEX_TYPE.all, EPI_MEASURE.infection),
            (AGG_INDEX_TYPE.death, EPI_MEASURE.death),
            (AGG_INDEX_TYPE.admission, EPI_MEASURE.admission),
            (AGG_INDEX_TYPE.case, EPI_MEASURE.case))
VACCINE_STATUS_MAP = ((AGG_INDEX_TYPE.unvaccinated, VACCINE_STATUS.unvaccinated),
                      (AGG_INDEX_TYPE.vaccinated, VACCINE_STATUS.vaccinated),
                      (AGG_INDEX_TYPE.booster, VACCINE_STATUS.booster))
EPI_MEASURE_MAP = ((TRACKING_COMPARTMENT.Infection, EPI_MEASURE.infection),
                   (TRACKING_COMPARTMENT.Death, EPI_MEASURE.death),
                   (TRACKING_COMPARTMENT.Admission, EPI_MEASURE.admission),
                   (TRACKING_COMPARTMENT.Case, EPI_MEASURE.case))


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
                    if vaccine_status == VACCINE_STATUS.unvaccinated:
                        dy[TRACKING_COMPARTMENTS[tc, variant_from, VARIANT_GROUP.all, agg_vaccine_status]] += measure
                        dy[TRACKING_COMPARTMENTS[tc, variant_from, variant_to, agg_vaccine_status]] += measure
                dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.all]] += measure
                dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, variant_to, AGG_INDEX_TYPE.all]] += measure

    tc_vax = TRACKING_COMPARTMENT.Vaccination
    tc_boost = TRACKING_COMPARTMENT.Booster
    tc_effs = TRACKING_COMPARTMENT.EffectiveSusceptible
    for variant in VARIANT:
        dy[TRACKING_COMPARTMENTS[tc_vax, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.unvaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.unvaccinated, variant],
                           COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.vaccinated, variant]]
        )
        dy[TRACKING_COMPARTMENTS[tc_boost, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.vaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.vaccinated, variant],
                           COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.booster, variant]]
        )

        for variant_from, vaccine_status in cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS))):
            dy[TRACKING_COMPARTMENTS[tc_effs, VARIANT_GROUP.all, variant, AGG_INDEX_TYPE.all]] += (
                effective_susceptible[EFFECTIVE_SUSCEPTIBLE[vaccine_status, variant_from, variant]]
            )

    if DEBUG:
        assert np.all(np.isfinite(dy))
    return dy

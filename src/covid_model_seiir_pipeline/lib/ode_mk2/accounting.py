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
                                  group_dy: np.ndarray,
                                  new_e: np.ndarray,
                                  effective_susceptible: np.ndarray,
                                  beta: np.ndarray,
                                  group_outcomes: np.ndarray,
                                  transition_map: np.ndarray) -> np.ndarray:
    for agg_idx, epi_idx in BETA_MAP:
        beta_idx = TRACKING_COMPARTMENTS[TRACKING_COMPARTMENT_TYPE.Beta, VARIANT.none, VARIANT.none, agg_idx]
        group_dy[beta_idx] = beta[epi_idx]

    for tc, epi_measure in EPI_MEASURE_MAP[:1]:
        for variant_from, variant_to in cartesian_product((np.array(VARIANT), np.array(VARIANT))):
            for agg_vaccine_status, vaccine_status in VACCINE_STATUS_MAP:
                if epi_measure == EPI_MEASURE.infection:

                    if variant_from == VARIANT.none:
                        if vaccine_status == VACCINE_STATUS.unvaccinated:
                            group_dy[TRACKING_COMPARTMENTS[tc, variant_from, VARIANT_GROUP.all, vaccine_status]] += (
                                new_e[NEW_E[vaccine_status, variant_from, variant_to]]
                            )
                        group_dy[TRACKING_COMPARTMENTS[tc, variant_from, VARIANT_GROUP.all, AGG_INDEX_TYPE.all]] += (
                            new_e[NEW_E[vaccine_status, variant_from, variant_to]]
                        )
                    group_dy[
                        TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, variant_to, AGG_INDEX_TYPE.all]] += (
                        new_e[NEW_E[vaccine_status, variant_from, variant_to]]
                    )
                    group_dy[TRACKING_COMPARTMENTS[tc, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.all]] += (
                        new_e[NEW_E[vaccine_status, variant_from, variant_to]]
                    )
    for tc, epi_measure in EPI_MEASURE_MAP[1:]:
        group_dy[TRACKING_COMPARTMENTS[tc, VARIANT.none, VARIANT_GROUP.all, AGG_INDEX_TYPE.unvaccinated]] += (
            group_outcomes[epi_measure]
        )
    tc_vax = TRACKING_COMPARTMENT.Vaccination
    tc_boost = TRACKING_COMPARTMENT.Booster
    tc_effs = TRACKING_COMPARTMENT.EffectiveSusceptible
    for variant in VARIANT:
        group_dy[TRACKING_COMPARTMENTS[tc_vax, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.unvaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.unvaccinated, variant],
                           COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.vaccinated, variant]]
        )
        group_dy[TRACKING_COMPARTMENTS[tc_boost, VARIANT_GROUP.all, VARIANT_GROUP.all, AGG_INDEX_TYPE.vaccinated]] += (
            transition_map[COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.vaccinated, variant],
                           COMPARTMENTS[COMPARTMENT.S, VACCINE_STATUS.booster, variant]]
        )
        for variant_from, vaccine_status in cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS))):
            group_dy[TRACKING_COMPARTMENTS[tc_effs, VARIANT_GROUP.all, variant, AGG_INDEX_TYPE.all]] += (
                effective_susceptible[EFFECTIVE_SUSCEPTIBLE[vaccine_status, variant_from, variant]]
            )

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

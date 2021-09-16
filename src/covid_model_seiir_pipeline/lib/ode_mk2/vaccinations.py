import numpy as np
import numba

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    BASE_COMPARTMENT,
    VARIANT,
    PROTECTION_STATUS,
    VACCINE_TYPE,
    VACCINATION_STATUS,
    AGG_OTHER,
    # Indexing arrays
    COMPARTMENTS,
    # Compartment groups
    CG_TOTAL,
    # Debug flag
    DEBUG,
)


@numba.njit
def allocate(
    group_y: np.ndarray,
    group_vaccines: np.ndarray,
    force_of_infection: np.ndarray,
) -> np.ndarray:
    # Allocate our output space.
    offset = min(VACCINE_TYPE)
    vaccines_out = np.zeros((group_y.size, max(VACCINE_TYPE) + 1))

    available_vaccines = group_vaccines.sum()
    vaccine_eligible = group_y[CG_TOTAL[AGG_OTHER.total]].sum()

    # Don't vaccinate if no vaccines to deliver or there is no-one to vaccinate.
    if available_vaccines == 0 or vaccine_eligible == 0:
        return vaccines_out

    for protection_status in PROTECTION_STATUS:
        s_index = COMPARTMENTS[BASE_COMPARTMENT.S, protection_status, VACCINATION_STATUS.unvaccinated]
        s = group_y[s_index]
        new_e_from_s = s * force_of_infection.sum()

        expected_vaccinations_from_s_by_type = s / vaccine_eligible * group_vaccines
        expected_vaccinations_from_s = expected_vaccinations_from_s_by_type.sum()
        # Transmission takes precedence over vaccination
        total_vaccinations_from_s = min(s - new_e_from_s, expected_vaccinations_from_s)
        assert total_vaccinations_from_s > 0
        vaccine_scale = math.safe_divide(total_vaccinations_from_s, expected_vaccinations_from_s)
        total_vaccinations_from_s_by_type = vaccine_scale * expected_vaccinations_from_s_by_type

        for v_index in VACCINE_TYPE:
            # Vaccines can't remove protection
            to_v_index = max(v_index, protection_status)
            vaccines_out[s_index, to_v_index] += total_vaccinations_from_s_by_type[v_index - offset]

    for variant in VARIANT:
        r_index = COMPARTMENTS[BASE_COMPARTMENT.R, variant, VACCINATION_STATUS.unvaccinated]
        r = group_y[r_index]
        # TODO:
        #waned_index = NATURAL_IMMUNITY_WANED[(variant,)]
        #waned_from_r = natural_immunity_waned[waned_index]
        waned_from_r = 0.

        expected_vaccinations_from_r_by_type = r / vaccine_eligible * group_vaccines
        expected_vaccinations_from_r = expected_vaccinations_from_r_by_type.sum()
        # Waning takes precedence over vaccination
        total_vaccinations_from_r = min(r - waned_from_r, expected_vaccinations_from_r)
        assert total_vaccinations_from_r > 0
        vaccine_scale = math.safe_divide(total_vaccinations_from_r, expected_vaccinations_from_r)
        total_vaccinations_from_r_by_type = vaccine_scale * expected_vaccinations_from_r_by_type

        # Non-immunizing vaccines all have the same impact (move to newly vaccinated).
        for v_index in VACCINE_TYPE:
            to_v_index = VACCINE_TYPE.unprotected if v_index < VACCINE_TYPE.non_escape_immune else v_index
            vaccines_out[r_index, to_v_index] += total_vaccinations_from_r_by_type[v_index - offset]

    if DEBUG:
        assert np.all(np.isfinite(vaccines_out))
        assert np.all(vaccines_out >= 0)

    return vaccines_out

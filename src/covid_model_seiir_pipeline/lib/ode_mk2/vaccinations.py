import numpy as np
import numba

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT,
    COMPARTMENTS,
    PROTECTION_STATUS,
    VACCINE_TYPES,
    COMPARTMENT_GROUPS,
    DEBUG,
)


@numba.njit
def allocate(
    group_y: np.ndarray,
    group_vaccines: np.ndarray,
    force_of_infection: np.ndarray,
) -> np.ndarray:
    # Allocate our output space.
    vaccines_out = np.zeros((group_y.size, group_vaccines.size))

    available_vaccines = group_vaccines.sum()
    vaccine_eligible = group_y[COMPARTMENT_GROUPS[('N', 'vaccine_eligible')]].sum()

    # Don't vaccinate if no vaccines to deliver or there is no-one to vaccinate.
    if available_vaccines == 0 or vaccine_eligible == 0:
        return vaccines_out

    for protection_status in PROTECTION_STATUS:
        s_index = COMPARTMENTS[('S', protection_status, 'unvaccinated')]
        s = group_y[s_index]
        new_e_from_s = s * force_of_infection.sum()

        expected_vaccinations_from_s_by_type = s / vaccine_eligible * group_vaccines
        expected_vaccinations_from_s = expected_vaccinations_from_s_by_type.sum()
        # Transmission takes precedence over vaccination
        total_vaccinations_from_s = min(s - new_e_from_s, expected_vaccinations_from_s)
        vaccine_scale = math.safe_divide(total_vaccinations_from_s, expected_vaccinations_from_s)
        total_vaccinations_from_s_by_type = vaccine_scale * expected_vaccinations_from_s_by_type

        for v_index in VACCINE_TYPES.values():
            # Vaccines can't remove protection
            to_v_index = max(v_index, VACCINE_TYPES[(protection_status,)])
            vaccines_out[s_index, to_v_index] += total_vaccinations_from_s_by_type[v_index]

    for variant in VARIANT:
        r_index = COMPARTMENTS[('R', variant, 'unvaccinated')]
        r = group_y[r_index]
        # TODO:
        #waned_index = NATURAL_IMMUNITY_WANED[(variant,)]
        #waned_from_r = natural_immunity_waned[waned_index]
        waned_from_r = 0.

        expected_vaccinations_from_r_by_type = r / vaccine_eligible * group_vaccines
        expected_vaccinations_from_r = expected_vaccinations_from_r_by_type.sum()
        # Waning takes precedence over vaccination
        total_vaccinations_from_r = min(r - waned_from_r, expected_vaccinations_from_r)
        vaccine_scale = math.safe_divide(total_vaccinations_from_r, expected_vaccinations_from_r)
        total_vaccinations_from_r_by_type = vaccine_scale * expected_vaccinations_from_r_by_type

        # Non-immunizing vaccines all have the same impact (move to newly vaccinated).
        unprotected_index = VACCINE_TYPES[('unprotected',)]
        min_index = VACCINE_TYPES[('non_escape_immune',)]
        for v_index in VACCINE_TYPES.values():
            to_v_index = unprotected_index if v_index < min_index else v_index
            vaccines_out[r_index, to_v_index] += total_vaccinations_from_r_by_type[v_index]

    if DEBUG:
        assert np.all(np.isfinite(vaccines_out))
        assert np.all(vaccines_out >= 0)

    return vaccines_out

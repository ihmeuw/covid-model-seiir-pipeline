"""Subroutines to initialize escape variant invasion."""
import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    COMPARTMENT,
    VARIANT,
    VARIANT_GROUP,
    BASE_PARAMETER,
    VARIANT_PARAMETER,
    VACCINE_STATUS,
    # Indexing arrays
    COMPARTMENTS,
    PARAMETERS,
    AGGREGATES,
    # Debug flag
    DEBUG,
)


@numba.njit
def maybe_invade(group_y: np.ndarray, group_dy: np.ndarray,
                 aggregates: np.ndarray, params: np.ndarray) -> np.ndarray:
    alpha = params[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all]]
    pi = params[PARAMETERS[BASE_PARAMETER.pi, VARIANT_GROUP.all]]
    from_compartment = COMPARTMENTS[COMPARTMENT.S, VARIANT.none, VACCINE_STATUS.unvaccinated]

    for variant in VARIANT:
        no_variant_present = params[PARAMETERS[VARIANT_PARAMETER.rho, variant]] < 0.01
        already_invaded = aggregates[AGGREGATES[COMPARTMENT.I, variant]] > 0.0
        # Short circuit if we don't have variant invasion this step
        if no_variant_present or already_invaded:
            continue
            
        # Shift at least 1 person to the escape variants if we can
        min_invasion = 1
        # Cap the the invasion so we don't take everyone. The handles corner cases
        # where variants invade just as vaccination is starting.
        max_invasion = 1 / (2*len(VARIANT)) * group_y[from_compartment]
        exposed = 0.
        for vaccination_status in VACCINE_STATUS:
            exposed += group_y[COMPARTMENTS[COMPARTMENT.E, VARIANT.ancestral, vaccination_status]]
        delta = min(max(pi * exposed, min_invasion), max_invasion)

        # Set the boundary condition so that the initial beta for the escape
        # variant starts at 5 (for consistency with ancestral type invasion).
        group_dy[from_compartment] -= delta + (delta / 5) ** (1 / alpha)
        group_dy[COMPARTMENTS[COMPARTMENT.E, variant, VACCINE_STATUS.unvaccinated]] += delta
        group_dy[COMPARTMENTS[COMPARTMENT.I, variant, VACCINE_STATUS.unvaccinated]] += (delta / 5) ** (1 / alpha)

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

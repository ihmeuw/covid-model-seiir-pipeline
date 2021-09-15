"""Subroutines to initialize escape variant invasion."""
import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    BASE_COMPARTMENT,
    VARIANT,
    VARIANT_GROUP,
    BASE_PARAMETER,
    VARIANT_PARAMETER,
    SUSCEPTIBLE_TYPE,
    VACCINATION_STATUS,
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
    from_compartment = COMPARTMENTS[BASE_COMPARTMENT.S, SUSCEPTIBLE_TYPE.unprotected, VACCINATION_STATUS.unvaccinated]

    for variant in VARIANT:
        no_variant_present = params[PARAMETERS[VARIANT_PARAMETER.rho, variant]] < 0.01
        already_invaded = aggregates[AGGREGATES[BASE_COMPARTMENT.I, variant]] > 0.0
        # Short circuit if we don't have variant invasion this step
        if no_variant_present or already_invaded:
            continue
        # Shift at least 1 person to the escape variants if we can
        min_invasion = 1
        # Cap the the invasion so we don't take everyone. The handles corner cases
        # where variants invade just as vaccination is starting.
        max_invasion = 1 / (2*len(VARIANT)) * group_y[from_compartment]
        exposed = group_y[COMPARTMENTS[BASE_COMPARTMENT.E, VARIANT.ancestral]]
        delta = min(max(pi * group_y[exposed], min_invasion), max_invasion)

        # Set the boundary condition so that the initial beta for the escape
        # variant starts at 5 (for consistency with ancestral type invasion).
        group_dy[from_compartment] -= delta + (delta / 5) ** (1 / alpha)
        group_dy[COMPARTMENTS[BASE_COMPARTMENT.E, variant]] += delta
        group_dy[COMPARTMENTS[BASE_COMPARTMENT.E, variant]] += (delta / 5) ** (1 / alpha)

    if DEBUG:
        assert np.all(np.isfinite(group_dy))

    return group_dy

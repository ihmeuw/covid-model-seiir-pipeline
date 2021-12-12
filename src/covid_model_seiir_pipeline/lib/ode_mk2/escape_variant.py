"""Subroutines to initialize escape variant invasion."""
import numba
import numpy as np

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Indexing tuples
    RISK_GROUP,
    COMPARTMENT,
    VARIANT,
    VARIANT_GROUP,
    BASE_PARAMETER,
    VARIANT_PARAMETER,
    VACCINE_STATUS,
    EPI_MEASURE,
    # Indexing arrays
    COMPARTMENTS,
    PARAMETERS,
    AGGREGATES,
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
    Printer
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    utils,
)


@numba.njit
def maybe_invade(t: float,
                 y: np.ndarray,
                 params: np.ndarray) -> np.ndarray:
    alpha = params[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all, EPI_MEASURE.infection]]
    pi = params[PARAMETERS[BASE_PARAMETER.pi, VARIANT_GROUP.all, EPI_MEASURE.infection]]

    total_susceptible = 0.
    total_exposed = 0.
    for risk_group in RISK_GROUP:
        group_y = utils.subset_risk_group(y, risk_group)
        for variant, vaccine_status in utils.cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS))):
            total_susceptible += group_y[COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant]]
            total_exposed += (
                group_y[COMPARTMENTS[COMPARTMENT.E, vaccine_status, variant]]
                + group_y[COMPARTMENTS[COMPARTMENT.I, vaccine_status, variant]]
            )

    # At least 1 person if more than 1 person available
    min_invasion = 1.0
    # At most .5% of the total susceptible population
    max_invasion = 0.005 * total_susceptible
    delta = max(min(pi * total_exposed, max_invasion), min_invasion)

    for variant_to in VARIANT:
        no_variant_present = params[PARAMETERS[VARIANT_PARAMETER.rho, variant_to, EPI_MEASURE.infection]] < 0.01
        already_invaded = False
        for vaccine_status in VACCINE_STATUS:
            if y[COMPARTMENTS[COMPARTMENT.I, vaccine_status, variant_to]] > 0.0:
                already_invaded = True

        # Short circuit if we don't have variant invasion this step
        if no_variant_present or already_invaded:
            continue

        for risk_group in RISK_GROUP:
            group_y = utils.subset_risk_group(y, risk_group)
            for vaccine_status in VACCINE_STATUS:
                e_idx = COMPARTMENTS[COMPARTMENT.E, vaccine_status, variant_to]
                i_idx = COMPARTMENTS[COMPARTMENT.I, vaccine_status, variant_to]
                for variant_from in VARIANT:
                    from_compartment = COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant_from]
                    compartment_delta = group_y[from_compartment] / total_susceptible * delta
                    group_y[from_compartment] -= compartment_delta + (compartment_delta / 5) ** (1 / alpha)
                    group_y[e_idx] += compartment_delta
                    group_y[i_idx] += (compartment_delta / 5) ** (1 / alpha)
    return y

import numba
import numpy as np
from typing import Tuple

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    COMPARTMENT_TYPE,
    RISK_GROUP,
    VARIANT,
    VARIANT_GROUP,
    VACCINE_STATUS,
    COMPARTMENT,
    BASE_PARAMETER,
    VARIANT_PARAMETER,

    AGGREGATES,
    PARAMETERS,
    PHI,
    NEW_E,
    COMPARTMENTS,

    DEBUG,
)


@numba.njit
def make_aggregates(y: np.ndarray) -> np.ndarray:
    aggregates = np.zeros(AGGREGATES.max() + 1)
    for group_y in np.split(y, len(RISK_GROUP)):
        # Total population
        aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]] += group_y[:COMPARTMENTS.max() + 1].sum()
        # Infectious by variant
        for variant in VARIANT:
            aggregates[AGGREGATES[COMPARTMENT.I, variant]] += group_y[COMPARTMENTS[COMPARTMENT.I, variant]].sum()

    if DEBUG:
        assert np.all(np.isfinite(aggregates))

    return aggregates


@numba.njit
def make_new_e(t: float,
               group_y: np.ndarray,
               parameters: np.ndarray,
               aggregates: np.ndarray,
               phis: np.ndarray,
               forecast: bool) -> Tuple[np.ndarray, np.ndarray]:

    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]]
    alpha = parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all]]
    new_e = np.zeros(NEW_E.max() + 1)
    effective_susceptible = np.zeros(len(VARIANT) + 1)

    if forecast:
        beta = parameters[PARAMETERS[VARIANT_PARAMETER.beta, VARIANT_GROUP.all]]
        for variant_x in VARIANT:
            infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_x]]**alpha
            for vaccine_status in VACCINE_STATUS:
                susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_x, vaccine_status]]
                for variant_y in VARIANT:
                    kappa = parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant_y]]
                    eta = parameters[PARAMETERS[VARIANT_PARAMETER.eta, variant_y]]
                    phi = phis[PHI[variant_y, variant_x]]
                    s_effective = eta * phi * susceptible
                    effective_susceptible[variant_y] += s_effective
                    new_e[NEW_E[variant_x, variant_y, vaccine_status]] += (
                        beta * kappa * s_effective * infectious / n_total
                    )

    else:
        new_e_total = parameters[PARAMETERS[BASE_PARAMETER.new_e, VARIANT_GROUP.all]]
        total_weight = 0.
        for variant_x in VARIANT:
            infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_x]] ** alpha
            for vaccine_status in VACCINE_STATUS:
                susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_x, vaccine_status]]
                for variant_y in VARIANT:
                    kappa = parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant_y]]
                    eta = parameters[PARAMETERS[VARIANT_PARAMETER.eta, variant_y]]
                    phi = phis[PHI[variant_y, variant_x]]
                    s_effective = eta * phi * susceptible
                    effective_susceptible[variant_y] += s_effective
                    variant_weight = kappa * s_effective * infectious
                    total_weight += variant_weight
                    new_e[NEW_E[variant_x, variant_y, vaccine_status]] += (
                        variant_weight * new_e_total
                    )
        new_e /= total_weight

    if DEBUG:
        assert np.all(np.isfinite(new_e))

    return new_e, effective_susceptible

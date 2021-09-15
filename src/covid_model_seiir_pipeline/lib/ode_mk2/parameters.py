import numba
import numpy as np
from typing import Tuple

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    # Base indexing tuples
    COMPARTMENT_TYPE,
    # Indexing tuples
    RISK_GROUP,
    BASE_COMPARTMENT,
    BASE_PARAMETER,
    VARIANT_PARAMETER,
    VARIANT,
    VARIANT_GROUP,
    AGG_VACCINATION_STATUS,
    AGG_OTHER,
    # Indexing arrays
    PARAMETERS,
    AGGREGATES,
    # Compartment groups
    CG_SUSCEPTIBLE,
    CG_EXPOSED,
    CG_INFECTIOUS,
    CG_REMOVED,
    CG_TOTAL,
    # Debug flag
    DEBUG,
)


@numba.njit
def make_aggregates(y: np.ndarray) -> np.ndarray:
    aggregates = np.zeros(AGGREGATES.max() + 1)
    for group_y in np.split(y, len(RISK_GROUP)):
        for compartment, group in zip(BASE_COMPARTMENT, (CG_SUSCEPTIBLE, CG_EXPOSED, CG_INFECTIOUS, CG_REMOVED)):
            for variant in VARIANT:
                aggregates[AGGREGATES[compartment, variant]] = group_y[group[variant]].sum()
        for vaccination_status in AGG_VACCINATION_STATUS:
            n_vax_status = group_y[CG_TOTAL[vaccination_status]].sum()
            aggregates[AGGREGATES[COMPARTMENT_TYPE.N, vaccination_status]] = n_vax_status
            aggregates[AGGREGATES[COMPARTMENT_TYPE.N, AGG_OTHER.total]] += n_vax_status
    if DEBUG:
        assert np.all(np.isfinite(aggregates))

    return aggregates


@numba.njit
def normalize_parameters(input_parameters: np.ndarray,
                         aggregates: np.ndarray,
                         forecast: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    force_of_infection = np.zeros(max(VARIANT)+1)

    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, AGG_OTHER.total]]
    params = input_parameters[:PARAMETERS.max()+1]
    vaccines = input_parameters[PARAMETERS.max()+1:]

    alpha = input_parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all]]

    if forecast:
        for variant in VARIANT:
            beta = input_parameters[PARAMETERS[VARIANT_PARAMETER.beta, variant]]
            infectious = aggregates[AGGREGATES[BASE_COMPARTMENT.I, variant]]
            force_of_infection[variant] = beta * infectious**alpha / n_total
    else:
        new_e_total = input_parameters[PARAMETERS[BASE_PARAMETER.new_e, VARIANT_GROUP.all]]
        total_weight = 0.
        new_e = np.zeros(max(VARIANT)+1)
        for variant in VARIANT:
            kappa = input_parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant]]
            susceptible = aggregates[AGGREGATES[BASE_COMPARTMENT.S, variant]]
            infectious = aggregates[AGGREGATES[BASE_COMPARTMENT.I, variant]]

            variant_weight = kappa * susceptible * infectious**alpha
            new_e[variant] = new_e_total * variant_weight
            total_weight += variant_weight
        new_e /= total_weight
        for variant in VARIANT:
            susceptible = aggregates[AGGREGATES[BASE_COMPARTMENT.S, variant]]
            force_of_infection[variant] = math.safe_divide(new_e[variant], susceptible)

    if DEBUG:
        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(vaccines))
        assert np.all(np.isfinite(force_of_infection))

    return params, vaccines, force_of_infection

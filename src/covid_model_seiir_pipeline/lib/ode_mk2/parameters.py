import numba
import numpy as np
from typing import Tuple

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP,
    AGGREGATES,
    PARAMETERS,
    NEW_E,
    NATURAL_IMMUNITY_WANED,
    VACCINE_IMMUNITY_WANED,
    FORCE_OF_INFECTION,
    COMPARTMENT_GROUPS,
    DEBUG,
)


@numba.njit
def make_aggregates(y: np.ndarray) -> np.ndarray:
    aggregates = np.zeros(len(AGGREGATES))
    for group_y in np.split(y, len(RISK_GROUP)):
        for key, agg_idx in AGGREGATES:
            aggregates[agg_idx] = group_y[COMPARTMENT_GROUPS[key]].sum()

    if DEBUG:
        assert np.all(np.isfinite(aggregates))

    return aggregates


@numba.njit
def normalize_parameters(input_parameters: np.ndarray,
                         aggregates: np.ndarray,
                         forecast: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray]:
    new_e = np.zeros(len(NEW_E))
    force_of_infection = np.zeros(len(FORCE_OF_INFECTION))
    natural_immunity_waned = np.zeros(len(NATURAL_IMMUNITY_WANED))
    vaccine_immunity_waned = np.zeros(len(VACCINE_IMMUNITY_WANED))

    n_total = aggregates[AGGREGATES[('N', 'total')]]
    params = input_parameters[:PARAMETERS[('new_e', 'total')]]
    vaccines = input_parameters[PARAMETERS[('new_e', 'total')]+1:]

    if forecast:
        for variant, new_e_idx in NEW_E.items():
            variant = variant[0]  # Key is a 1-element tuple
            alpha = input_parameters[PARAMETERS[('alpha', variant)]]
            beta = input_parameters[PARAMETERS[('beta', variant)]]

            susceptible = aggregates[AGGREGATES[('S', variant)]]
            infectious = aggregates[AGGREGATES[('I', variant)]]
            new_e[new_e_idx] = beta * susceptible * infectious**alpha / n_total
            # new_e and force_of_infection are identically indexed.
            force_of_infection[new_e_idx] = beta * infectious**alpha / n_total
    else:
        new_e_total = input_parameters[PARAMETERS[(new_e, 'total')]]
        total_weight = 0.
        for variant, new_e_idx in NEW_E.items():
            variant = variant[0]  # Key is a 1-element tuple
            alpha = input_parameters[PARAMETERS[('alpha', variant)]]
            kappa = input_parameters[PARAMETERS[('kappa', variant)]]

            susceptible = aggregates[AGGREGATES[('S', variant)]]
            infectious = aggregates[AGGREGATES[('I', variant)]]

            variant_weight = kappa * susceptible * infectious**alpha
            new_e[new_e_idx] = new_e_total * variant_weight
            total_weight += variant_weight
        new_e /= total_weight
        for variant, foi_idx in FORCE_OF_INFECTION.items():
            variant = variant[0]  # Key is a 1-element tuple
            susceptible = aggregates[AGGREGATES[('S', variant)]]
            # new_e and force_of_infection are identically indexed.
            force_of_infection[foi_idx] = math.safe_divide(new_e[foi_idx], susceptible)

    if DEBUG:
        assert np.all(np.isfinite(params))
        assert np.all(np.isfinite(vaccines))
        assert np.all(np.isfinite(new_e))
        assert np.all(np.isfinite(force_of_infection))

    return params, vaccines, new_e, force_of_infection, natural_immunity_waned, vaccine_immunity_waned

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
    CHI,
    ETA,
    NEW_E,
    EFFECTIVE_SUSCEPTIBLE,
    COMPARTMENTS,
)
from covid_model_seiir_pipeline.lib.ode_mk2.debug import (
    DEBUG,
    Printer
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
               etas: np.ndarray,
               chis: np.ndarray,
               forecast: bool) -> Tuple[np.ndarray, np.ndarray, float]:

    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]]
    alpha = parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all]]
    new_e = np.zeros(NEW_E.max() + 1)
    effective_susceptible = np.zeros(EFFECTIVE_SUSCEPTIBLE.max() + 1)

    if forecast:
        beta = parameters[PARAMETERS[BASE_PARAMETER.beta, VARIANT_GROUP.all]]
        for variant_to in VARIANT:            
            kappa = parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant_to]]
            infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]]**alpha            
            for variant_from in VARIANT:
                chi = chis[CHI[variant_from, variant_to]]            
                for vaccine_status in VACCINE_STATUS:
                    susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]]                    
                    eta = etas[ETA[vaccine_status, variant_to]]                    
                    s_effective = (1 - eta) * (1 - chi) * susceptible
                    effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_to, vaccine_status]] += s_effective
                    new_e[NEW_E[vaccine_status, variant_from, variant_to]] += (
                        beta * kappa * s_effective * infectious / n_total
                    )

    else:
        # TODO: Better than this
        group_pop = group_y[:COMPARTMENTS.max() + 1].sum()
        new_e_total = parameters[PARAMETERS[BASE_PARAMETER.new_e, VARIANT_GROUP.all]] * group_pop / n_total
        
        total_variant_weight = 0.        
        for variant_to in VARIANT:
            kappa = parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant_to]]
            infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]] ** alpha            
            for variant_from in VARIANT:
                chi = chis[CHI[variant_from, variant_to]]
                for vaccine_status in VACCINE_STATUS:
                    susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]]                    
                    eta = etas[ETA[vaccine_status, variant_to]]
                    s_effective = (1 - eta) * (1 - chi) * susceptible
                    if kappa > 0:
                        effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_to, vaccine_status]] += s_effective
                    variant_weight = kappa * s_effective * infectious
                    total_variant_weight += variant_weight                    
                    new_e[NEW_E[vaccine_status, variant_from, variant_to]] += (
                        variant_weight * new_e_total
                    )
        new_e = new_e / total_variant_weight
        beta = new_e_total * n_total / total_variant_weight
    if DEBUG:
        assert np.all(np.isfinite(new_e))

    return new_e, effective_susceptible, beta

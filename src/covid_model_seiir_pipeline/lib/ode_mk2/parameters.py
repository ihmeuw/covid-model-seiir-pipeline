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
            for vaccine_status in VACCINE_STATUS:
                aggregates[AGGREGATES[COMPARTMENT.I, variant]] += group_y[COMPARTMENTS[COMPARTMENT.I, variant, vaccine_status]]

    if DEBUG:
        assert np.all(np.isfinite(aggregates))
    
    return aggregates


@numba.njit
def make_new_e(t: float,
               y: np.ndarray,
               parameters: np.ndarray,               
               aggregates: np.ndarray,
               etas: np.ndarray,
               chis: np.ndarray,
               forecast: bool) -> Tuple[np.ndarray, np.ndarray, float]:

    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]]
    alpha = parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all]]
    new_e = np.zeros(2*(NEW_E.max() + 1))
    effective_susceptible = np.zeros(2*(EFFECTIVE_SUSCEPTIBLE.max() + 1))    

    if forecast:
        beta = parameters[PARAMETERS[BASE_PARAMETER.beta, VARIANT_GROUP.all]]        
        
        for risk_group in RISK_GROUP:
            group_y = subset(y, risk_group)
            group_etas = subset(etas, risk_group)
            group_chis = subset(chis, risk_group)
            group_new_e = subset(new_e, risk_group)
            group_effective_susceptible = subset(effective_susceptible, risk_group)
            
            for variant_to in VARIANT:            
                kappa = parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant_to]]
                infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]]**alpha
                
                for variant_from in VARIANT:
                    chi = group_chis[CHI[variant_from, variant_to]]
                    
                    for vaccine_status in VACCINE_STATUS:
                        susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]]                    
                        eta = group_etas[ETA[vaccine_status, variant_to]]
                        s_effective = (1 - eta) * (1 - chi) * susceptible
                        if kappa > 0:
                            group_effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_to, vaccine_status]] += s_effective
                        group_new_e[NEW_E[vaccine_status, variant_from, variant_to]] += (
                            beta * kappa * s_effective * infectious / n_total
                        )

    else:        
        new_e_total = parameters[PARAMETERS[BASE_PARAMETER.new_e, VARIANT_GROUP.all]]
        total_variant_weight = 0.        
        
        for risk_group in RISK_GROUP:
            group_y = subset(y, risk_group)
            group_etas = subset(etas, risk_group)
            group_chis = subset(chis, risk_group)
            group_new_e = subset(new_e, risk_group)
            group_effective_susceptible = subset(effective_susceptible, risk_group)
        
            for variant_to in VARIANT:
                kappa = parameters[PARAMETERS[VARIANT_PARAMETER.kappa, variant_to]]
                infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]] ** alpha            
                
                for variant_from in VARIANT:
                    chi = group_chis[CHI[variant_from, variant_to]]
                    
                    for vaccine_status in VACCINE_STATUS:
                        susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]]                    
                        eta = group_etas[ETA[vaccine_status, variant_to]]
                        s_effective = (1 - eta) * (1 - chi) * susceptible
                        if kappa > 0:
                            group_effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_to, vaccine_status]] += s_effective
                        
                        variant_weight = kappa * s_effective * infectious
                        total_variant_weight += variant_weight
                        group_new_e[NEW_E[vaccine_status, variant_from, variant_to]] += (
                            variant_weight * new_e_total
                        )        
        new_e = new_e / total_variant_weight
        beta = new_e_total * n_total / total_variant_weight
    if DEBUG:
        assert np.all(np.isfinite(new_e))
    
    return new_e, effective_susceptible, beta


@numba.njit
def subset(x: np.ndarray, risk_group: int):
    x_size = x.size // len(RISK_GROUP)
    group_start = risk_group * x_size
    group_end = (risk_group + 1) * x_size
    return x[group_start:group_end]

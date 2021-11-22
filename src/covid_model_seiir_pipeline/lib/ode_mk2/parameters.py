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
    EPI_PARAMETER,
    VARIANT_PARAMETER,
    EPI_VARIANT_PARAMETER,
    EPI_MEASURE,
    REPORTED_EPI_MEASURE,

    AGGREGATES,
    PARAMETERS,
    BASE_RATES,
    RATES,
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
from covid_model_seiir_pipeline.lib.ode_mk2.utils import (
    subset_risk_group,
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
               base_rates: np.ndarray,
               aggregates: np.ndarray,
               etas: np.ndarray,
               chis: np.ndarray,
               forecast: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]]
    alpha = parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all, EPI_MEASURE.infection]]
    new_e = np.zeros(2*(NEW_E.max() + 1))
    effective_susceptible = np.zeros(2*(EFFECTIVE_SUSCEPTIBLE.max() + 1))    
    rates = np.zeros(2*(RATES.max() + 1))

    if forecast:
        beta = parameters[PARAMETERS[BASE_PARAMETER.beta, VARIANT_GROUP.all, EPI_MEASURE.infection]]
        
        for risk_group in RISK_GROUP:
            group_y = subset_risk_group(y, risk_group)
            group_etas = subset_risk_group(etas, risk_group)
            group_chis = subset_risk_group(chis, risk_group)
            group_new_e = subset_risk_group(new_e, risk_group)
            group_effective_susceptible = subset_risk_group(effective_susceptible, risk_group)
            
            for variant_to in VARIANT:            
                kappa = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, EPI_MEASURE.infection]]
                infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]]**alpha
                
                for variant_from in VARIANT:
                    chi_infection = group_chis[CHI[variant_from, variant_to, EPI_MEASURE.infection]]
                    
                    for vaccine_status in VACCINE_STATUS:
                        susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]]
                        eta_infection = group_etas[ETA[vaccine_status, variant_to, EPI_MEASURE.infection]]
                        s_effective = (1 - eta_infection) * (1 - chi_infection) * susceptible
                        if kappa > 0:
                            group_effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_to, vaccine_status]] += s_effective
                        group_new_e[NEW_E[vaccine_status, variant_from, variant_to]] += (
                            beta * kappa * s_effective * infectious / n_total
                        )

    else:        
        total_variant_weight = np.zeros(max(EPI_MEASURE) + 1)
        betas = np.zeros(max(EPI_MEASURE) + 1)
        outcomes = np.zeros(2*(max(EPI_MEASURE) + 1))        
        
        for risk_group in RISK_GROUP:
            group_y = subset_risk_group(y, risk_group)
            group_base_rates = subset_risk_group(base_rates, risk_group)
            group_etas = subset_risk_group(etas, risk_group)
            group_chis = subset_risk_group(chis, risk_group)
            
            group_new_e = subset_risk_group(new_e, risk_group)
            group_rates = subset_risk_group(rates, risk_group)
            group_effective_susceptible = subset_risk_group(effective_susceptible, risk_group)
        
            for variant_to in VARIANT:
                kappa_infection = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, EPI_MEASURE.infection]]
                infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]] ** alpha
                
                for variant_from in VARIANT:
                    
                    chi_infection = group_chis[CHI[variant_from, variant_to, EPI_MEASURE.infection]]
                    
                    for vaccine_status in VACCINE_STATUS:
                        
                        susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, variant_from, vaccine_status]]
                        eta_infection = group_etas[ETA[vaccine_status, variant_to, EPI_MEASURE.infection]]
                        
                        s_effective = (1 - eta_infection) * (1 - chi_infection) * susceptible                                                
                        if kappa_infection > 0:
                            group_effective_susceptible[EFFECTIVE_SUSCEPTIBLE[variant_from, variant_to, vaccine_status]] += s_effective
                        
                        variant_weight = kappa_infection * s_effective * infectious / n_total
                        
                        for epi_measure in REPORTED_EPI_MEASURE:                            
                            base_rate = group_base_rates[BASE_RATES[epi_measure]]                            
                            kappa = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, epi_measure]]
                            
                            chi = group_chis[CHI[variant_from, variant_to, epi_measure]]
                            eta = group_etas[ETA[vaccine_status, variant_to, epi_measure]]
                            
                            rate = kappa * (1 - eta) * (1 - chi) * base_rate
                            total_variant_weight[epi_measure] += rate * variant_weight 
                            group_rates[RATES[epi_measure, variant_from, variant_to, vaccine_status]] = rate                            
                        
                        total_variant_weight[EPI_MEASURE.infection] += variant_weight
                        group_new_e[NEW_E[vaccine_status, variant_from, variant_to]] += variant_weight
                        
        beta_weight = 0.        
        for epi_measure in REPORTED_EPI_MEASURE:
            count = parameters[PARAMETERS[EPI_PARAMETER.count, VARIANT_GROUP.all, epi_measure]]
            weight = parameters[PARAMETERS[EPI_PARAMETER.weight, VARIANT_GROUP.all, epi_measure]]            
            if not np.isnan(count):                
                betas[epi_measure] = count / total_variant_weight[epi_measure]
                betas[EPI_MEASURE.infection] += betas[epi_measure] * weight
                beta_weight += weight
        betas[EPI_MEASURE.infection] /= beta_weight
        new_e = betas[EPI_MEASURE.infection] * new_e
        
        for risk_group in RISK_GROUP:
            group_new_e = subset_risk_group(new_e, risk_group)
            group_rates = subset_risk_group(rates, risk_group)            
            group_outcomes = subset_risk_group(outcomes, risk_group)
            
            naive_infections = group_new_e[NEW_E[VACCINE_STATUS.unvaccinated, VARIANT.none, VARIANT.ancestral]]
            group_outcomes[EPI_MEASURE.infection] = naive_infections
            for epi_measure in REPORTED_EPI_MEASURE:
                if betas[epi_measure] > 0.:
                    r = group_rates[RATES[epi_measure, VARIANT.none, VARIANT.ancestral, VACCINE_STATUS.unvaccinated]]
                    group_outcomes[epi_measure] += r * naive_infections * betas[epi_measure] / betas[EPI_MEASURE.infection]
                    
        
    if DEBUG:
        assert np.all(np.isfinite(new_e))
        assert np.all(np.isfinite(effective_susceptible))
        assert np.all(np.isfinite(betas))
        assert np.all(np.isfinite(outcomes))
    
    return new_e, effective_susceptible, betas, outcomes




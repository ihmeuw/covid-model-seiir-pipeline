import numba
import numpy as np
from typing import Tuple

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    TOMBSTONE,

    COMPARTMENT_TYPE,
    RISK_GROUP,
    VARIANT,
    VARIANT_GROUP,
    VACCINE_STATUS,
    COMPARTMENT,
    BASE_PARAMETER,
    EPI_PARAMETER,
    EPI_VARIANT_PARAMETER,
    EPI_MEASURE,
    REPORTED_EPI_MEASURE,

    AGGREGATES,
    PARAMETERS,
    BASE_RATES,
    RATES,
    VARIANT_WEIGHTS,
    BETAS,
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
    cartesian_product,
    safe_divide
)


@numba.njit
def make_aggregates(y: np.ndarray) -> np.ndarray:
    aggregates = np.zeros(AGGREGATES.max() + 1)
    for group_y in np.split(y, len(RISK_GROUP)):
        # Total population
        aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]] += group_y[:COMPARTMENTS.max() + 1].sum()

        # Infectious by variant
        for variant, vaccine_status in cartesian_product((np.array(VARIANT), np.array(VACCINE_STATUS))):
            aggregates[AGGREGATES[COMPARTMENT.I, variant]] += (
                group_y[COMPARTMENTS[COMPARTMENT.I, vaccine_status, variant]]
            )

    if DEBUG:
        assert np.all(np.isfinite(aggregates))
        assert np.all(aggregates >= 0.)
    
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

    if forecast:
        raise
    else:
        total_variant_weight, variant_weight, effective_susceptible, rates = compute_intermediate_epi_parameters(
            t,
            y,
            parameters,
            base_rates,
            aggregates,
            etas,
            chis,
        )
        betas = compute_betas(parameters, total_variant_weight)
        new_e = betas[EPI_MEASURE.infection] * variant_weight

        for risk_group in RISK_GROUP:
            group_rates = subset_risk_group(rates, risk_group)
            for epi_measure in REPORTED_EPI_MEASURE:
                adjustment = safe_divide(betas[epi_measure], betas[EPI_MEASURE.infection])
                iteritems = cartesian_product((np.array(VARIANT), np.array(VARIANT), np.array(VACCINE_STATUS)))
                for variant_to, variant_from, vaccine_status in iteritems:
                    group_rates[RATES[vaccine_status, variant_from, variant_to, epi_measure]] *= adjustment
        
    if DEBUG:
        assert np.all(np.isfinite(new_e))
        assert np.all(new_e >= 0.)
        assert np.all(new_e < 1e7)
        assert np.all(np.isfinite(effective_susceptible))
        assert np.all(effective_susceptible >= 0.)
        assert np.all(np.isfinite(betas))
        assert np.all(betas >= 0.)
    
    return new_e, effective_susceptible, betas, rates


@numba.njit
def compute_intermediate_epi_parameters(t: float,
                                        y: np.ndarray,
                                        parameters: np.ndarray,
                                        base_rates: np.ndarray,
                                        aggregates: np.ndarray,
                                        etas: np.ndarray,
                                        chis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]]
    alpha = parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all, EPI_MEASURE.infection]]

    total_variant_weight = np.zeros(VARIANT_WEIGHTS.max() + 1)
    variant_weight = np.zeros(2 * (NEW_E.max() + 1))
    effective_susceptible = np.zeros(2 * (EFFECTIVE_SUSCEPTIBLE.max() + 1))
    rates = np.zeros(2 * (RATES.max() + 1))

    for risk_group in RISK_GROUP:
        group_y = subset_risk_group(y, risk_group)
        group_base_rates = subset_risk_group(base_rates, risk_group)
        group_etas = subset_risk_group(etas, risk_group)
        group_chis = subset_risk_group(chis, risk_group)

        group_rates = subset_risk_group(rates, risk_group)
        group_variant_weight = subset_risk_group(variant_weight, risk_group)
        group_effective_susceptible = subset_risk_group(effective_susceptible, risk_group)

        iteritems = cartesian_product((np.array(VARIANT), np.array(VARIANT), np.array(VACCINE_STATUS)))
        for variant_to, variant_from, vaccine_status in iteritems:
            susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant_from]]
            infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]] ** alpha
            kappa_infection = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, EPI_MEASURE.infection]]
            chi_infection = group_chis[CHI[variant_from, variant_to, EPI_MEASURE.infection]]
            eta_infection = group_etas[ETA[vaccine_status, variant_to, EPI_MEASURE.infection]]

            s_effective = (1 - eta_infection) * (1 - chi_infection) * susceptible
            weight = kappa_infection * s_effective * infectious / n_total

            for epi_measure in REPORTED_EPI_MEASURE:
                base_rate = group_base_rates[BASE_RATES[epi_measure]]
                kappa = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, epi_measure]]

                chi = group_chis[CHI[variant_from, variant_to, epi_measure]]
                eta = group_etas[ETA[vaccine_status, variant_to, epi_measure]]

                rate = kappa * (1 - eta) * (1 - chi) * base_rate
                total_variant_weight[epi_measure] += rate * weight
                group_rates[RATES[vaccine_status, variant_from, variant_to, epi_measure]] = rate

            total_variant_weight[EPI_MEASURE.infection] += weight
            group_variant_weight[NEW_E[vaccine_status, variant_from, variant_to]] += weight
            if kappa_infection > 0:
                eff_s_idx = EFFECTIVE_SUSCEPTIBLE[vaccine_status, variant_from, variant_to]
                group_effective_susceptible[eff_s_idx] += s_effective

    return total_variant_weight, variant_weight, effective_susceptible, rates


@numba.njit
def compute_betas(parameters: np.ndarray,
                  total_variant_weight: np.ndarray):
    betas = np.zeros(max(BETAS) + 1)
    beta_weight = 0.
    for epi_measure in REPORTED_EPI_MEASURE:
        count = parameters[PARAMETERS[EPI_PARAMETER.count, VARIANT_GROUP.all, epi_measure]]
        weight = parameters[PARAMETERS[EPI_PARAMETER.weight, VARIANT_GROUP.all, epi_measure]]
        if np.abs(count - TOMBSTONE) > 1e-8:
            betas[epi_measure] = count / total_variant_weight[epi_measure]
            betas[EPI_MEASURE.infection] += betas[epi_measure] * weight
            beta_weight += weight
    betas[EPI_MEASURE.infection] /= beta_weight
    return betas

import numba
import numpy as np
from typing import Tuple

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    TOMBSTONE,
    SYSTEM_TYPE,

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
    AGE_SCALARS,
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
def compute_intermediate_epi_parameters(t: float,
                                        y: np.ndarray,
                                        parameters: np.ndarray,
                                        age_scalars: np.ndarray,
                                        aggregates: np.ndarray,
                                        etas: np.ndarray,
                                        chis: np.ndarray,
                                        system_type: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n_total = aggregates[AGGREGATES[COMPARTMENT_TYPE.N, VARIANT_GROUP.total]]
    alpha = parameters[PARAMETERS[BASE_PARAMETER.alpha, VARIANT_GROUP.all, EPI_MEASURE.infection]]
    if system_type == SYSTEM_TYPE.rates_and_measures:
        beta = 1.0
    else:
        beta = parameters[PARAMETERS[BASE_PARAMETER.beta, VARIANT_GROUP.all, EPI_MEASURE.infection]]

    total_variant_weight = np.zeros(VARIANT_WEIGHTS.max() + 1)
    new_e = np.zeros(2 * (NEW_E.max() + 1))
    effective_susceptible = np.zeros(2 * (EFFECTIVE_SUSCEPTIBLE.max() + 1))
    rates = np.zeros(2 * (RATES.max() + 1))

    for risk_group in RISK_GROUP:
        group_y = subset_risk_group(y, risk_group)
        group_age_scalars = subset_risk_group(age_scalars, risk_group)
        group_etas = subset_risk_group(etas, risk_group)
        group_chis = subset_risk_group(chis, risk_group)

        group_rates = subset_risk_group(rates, risk_group)
        group_new_e = subset_risk_group(new_e, risk_group)
        group_effective_susceptible = subset_risk_group(effective_susceptible, risk_group)

        iteritems = cartesian_product((np.array(VARIANT), np.array(VARIANT), np.array(VACCINE_STATUS)))
        for variant_to, variant_from, vaccine_status in iteritems:
            susceptible = group_y[COMPARTMENTS[COMPARTMENT.S, vaccine_status, variant_from]]
            infectious = aggregates[AGGREGATES[COMPARTMENT.I, variant_to]] ** alpha
            kappa_infection = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, EPI_MEASURE.infection]]
            chi_infection = group_chis[CHI[variant_from, variant_to, EPI_MEASURE.infection]]
            eta_infection = group_etas[ETA[vaccine_status, variant_to, EPI_MEASURE.infection]]

            s_effective = (1 - eta_infection) * (1 - chi_infection) * susceptible
            new_e = beta * kappa_infection * s_effective * infectious / n_total

            for epi_measure in REPORTED_EPI_MEASURE:
                age_scalar = group_age_scalars[AGE_SCALARS[epi_measure]]
                if system_type == SYSTEM_TYPE.beta_and_measures:
                    base_rate = age_scalar
                else:
                    base_rate = age_scalar * parameters[PARAMETERS[EPI_PARAMETER.rate, VARIANT_GROUP.all, epi_measure]]

                kappa = parameters[PARAMETERS[EPI_VARIANT_PARAMETER.kappa, variant_to, epi_measure]]
                chi = group_chis[CHI[variant_from, variant_to, epi_measure]]
                eta = group_etas[ETA[vaccine_status, variant_to, epi_measure]]

                rate = kappa * (1 - eta) * (1 - chi) * base_rate
                total_variant_weight[VARIANT_WEIGHTS[epi_measure]] += rate * new_e
                group_rates[RATES[vaccine_status, variant_from, variant_to, epi_measure]] = rate

            total_variant_weight[VARIANT_WEIGHTS[EPI_MEASURE.infection]] += new_e
            group_new_e[NEW_E[vaccine_status, variant_from, variant_to]] += new_e
            group_effective_susceptible[EFFECTIVE_SUSCEPTIBLE[vaccine_status, variant_from, variant_to]] += s_effective

    if system_type == SYSTEM_TYPE.rates_and_measures:
        betas = compute_betas(parameters, total_variant_weight)
        rates = do_beta_rates_adjustment(rates, betas)
        new_e = betas[BETAS[EPI_MEASURE.infection]] * new_e
    elif system_type == SYSTEM_TYPE.beta_and_rates:
        betas = np.zeros(BETAS.max() + 1)
        betas[BETAS[EPI_MEASURE.infection]] = beta
    elif system_type == SYSTEM_TYPE.beta_and_measures:
        betas = np.zeros(BETAS.max() + 1)
        betas[BETAS[EPI_MEASURE.infection]] = beta
        base_rates = compute_base_rates(parameters, total_variant_weight)
        rates = do_rates_adjustment(rates, base_rates)
    else:
        raise

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
def compute_betas(parameters: np.ndarray,
                  total_variant_weight: np.ndarray):
    betas = np.zeros(max(BETAS) + 1)
    beta_weight = 0.
    for epi_measure in REPORTED_EPI_MEASURE:
        count = parameters[PARAMETERS[EPI_PARAMETER.count, VARIANT_GROUP.all, epi_measure]]
        weight = parameters[PARAMETERS[EPI_PARAMETER.weight, VARIANT_GROUP.all, epi_measure]]
        if np.abs(count - TOMBSTONE) > 1e-8:
            betas[BETAS[epi_measure]] = count / total_variant_weight[VARIANT_WEIGHTS[epi_measure]]
            betas[BETAS[EPI_MEASURE.infection]] += betas[BETAS[epi_measure]] * weight
            beta_weight += weight
    betas[BETAS[EPI_MEASURE.infection]] /= beta_weight
    return betas


@numba.njit
def do_beta_rates_adjustment(rates: np.ndarray, betas: np.ndarray) -> np.ndarray:
    for risk_group in RISK_GROUP:
        group_rates = subset_risk_group(rates, risk_group)
        for epi_measure in REPORTED_EPI_MEASURE:
            adjustment = safe_divide(betas[BETAS[epi_measure]],
                                     betas[BETAS[EPI_MEASURE.infection]])
            iteritems = cartesian_product((np.array(VARIANT), np.array(VARIANT), np.array(VACCINE_STATUS)))
            for variant_to, variant_from, vaccine_status in iteritems:
                group_rates[RATES[vaccine_status, variant_from, variant_to, epi_measure]] *= adjustment

    return rates


@numba.njit
def compute_base_rates(parameters: np.ndarray,
                       total_variant_weight: np.ndarray):
    base_rates = np.zeros(BASE_RATES.max() + 1)
    for epi_measure in REPORTED_EPI_MEASURE:
        count = parameters[PARAMETERS[EPI_PARAMETER.count, VARIANT_GROUP.all, epi_measure]]
        if np.abs(count - TOMBSTONE) > 1e-8:
            base_rates[BASE_RATES[epi_measure]] = count / total_variant_weight[VARIANT_WEIGHTS[epi_measure]]
    return base_rates


@numba.njit
def do_rates_adjustment(rates: np.ndarray, base_rates: np.ndarray) -> np.ndarray:
    for risk_group in RISK_GROUP:
        group_rates = subset_risk_group(rates, risk_group)
        for epi_measure in REPORTED_EPI_MEASURE:
            base_rate = base_rates[BASE_RATES[epi_measure]]
            iteritems = cartesian_product((np.array(VARIANT), np.array(VARIANT), np.array(VACCINE_STATUS)))
            for variant_to, variant_from, vaccine_status in iteritems:
                group_rates[RATES[vaccine_status, variant_from, variant_to, epi_measure]] *= base_rate

    return rates

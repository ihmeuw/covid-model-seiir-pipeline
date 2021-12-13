from collections import defaultdict
import itertools
from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd


from covid_model_seiir_pipeline.lib.ode_mk2.containers import Parameters
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    VACCINE_STATUS_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    PostprocessingParameters,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    compute_hospital_usage,
)


def compute_output_metrics(indices: Indices,
                           compartments: pd.DataFrame,
                           postprocessing_params: PostprocessingParameters,
                           model_parameters: Parameters,
                           hospital_parameters) -> Tuple[pd.DataFrame, pd.DataFrame]:
    system_metrics = compute_system_metrics(
        compartments,
    )

    # hospital_usage = compute_corrected_hospital_usage(
    #     admissions,
    #     hospital_parameters,
    #     postprocessing_params,
    # )

    r = compute_r(model_parameters,
                  system_metrics)

    output_metrics = pd.concat([
        infections,
        deaths,
        cases,
        hospital_usage.to_df(),
        r
    ], axis=1).set_index('observed', append=True)
    return system_metrics, output_metrics


def compute_system_metrics(compartments: pd.DataFrame) -> pd.DataFrame:
    total_pop = (compartments
                 .loc[:, [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], COMPARTMENTS_NAMES)]]
                 .sum(axis=1)
                 .rename('total_population'))
    compartments_diff = compartments.groupby('location_id').diff()

    infections = _make_measure(compartments_diff, 'Infection')
    deaths = _make_measure(compartments_diff, 'Death')
    admissions = _make_measure(compartments_diff, 'Admission')
    cases = _make_measure(compartments_diff, 'Case')
    susceptible = _make_susceptible(compartments_diff)
    immune = _make_immune(susceptible, total_pop)
    infectious = _make_infectious(compartments)
    vaccinations = _make_vaccinations(compartments)
    betas = compartments_diff.filter(like='Beta_none_none_all').mean(axis=1).rename('beta')
    force_of_infection = _make_force_of_infection(infections, susceptible)
    variant_prevalence = _make_variant_prevalence(infections)

    return pd.concat([
        infections,
        deaths,
        admissions,
        cases,
        susceptible,
        immune,
        infectious,
        vaccinations,
        betas,
        force_of_infection,
        variant_prevalence,
        total_pop,
    ], axis=1)


def _make_measure(compartments_diff: pd.DataFrame, measure: str) -> pd.DataFrame:
    # Ignore 'none'
    variant_names = VARIANT_NAMES[1:]
    data = defaultdict(lambda: pd.Series(0., index=compartments_diff.index))

    data['naive_unvaccinated'] = compartments_diff.filter(like=f'{measure}_none_all_unvaccinated').sum(axis=1)
    data['naive'] = compartments_diff.filter(like=f'{measure}_none_all_all').sum(axis=1)
    data['total'] = compartments_diff.filter(like=f'{measure}_all_all_all').sum(axis=1)

    for variant in variant_names:
        for risk_group in RISK_GROUP_NAMES:
            key = f'{measure}_all_{variant}_all_{risk_group}'
            data[variant] += compartments_diff[key]
            data[risk_group] += compartments_diff[key]

    infections = pd.concat([
        v.rename(f'modeled_{measure.lower()}s_{k}') for k, v in data.items()
    ], axis=1)

    return infections


def _make_susceptible(compartments_diff: pd.DataFrame) -> pd.DataFrame:
    variant_names = VARIANT_NAMES[1:]
    susceptible = defaultdict(lambda: pd.Series(0., index=compartments_diff.index))

    for variant in variant_names:
        for risk_group in RISK_GROUP_NAMES:
            key = f'EffectiveSusceptible_all_{variant}_all_{risk_group}'
            susceptible[variant] += compartments_diff[key]
            susceptible[risk_group] = np.maximum(susceptible[risk_group], compartments_diff[key])

    susceptible['total'] = sum([susceptible[risk_group] for risk_group in RISK_GROUP_NAMES])

    susceptible = pd.concat([
        v.rename(f'susceptible_{k}') for k, v in susceptible.items()
    ], axis=1)

    return susceptible


def _make_infectious(compartments: pd.DataFrame) -> pd.DataFrame:
    # Drop the first entry 'none'
    variant_names = VARIANT_NAMES[1:]
    infectious = defaultdict(lambda: pd.Series(0., index=compartments.index))

    for variant in variant_names:
        for vaccine_status in VACCINE_STATUS_NAMES:
            for risk_group in RISK_GROUP_NAMES:
                key = f'I_{vaccine_status}_{variant}_{risk_group}'
                infectious[variant] += compartments[key]
                infectious['total'] += compartments[key]

    infectious = pd.concat([
        v.rename(f'infectious_{k}') for k, v in infectious.items()
    ], axis=1)

    return infectious


def _make_immune(susceptible: pd.DataFrame, population: pd.Series) -> pd.DataFrame:
    # Drop the first entry 'none'
    variant_names = VARIANT_NAMES[1:]
    immune = defaultdict(lambda: pd.Series(0., index=susceptible.index))

    for variant in variant_names:
        immune[variant] = population - susceptible[f'susceptible_{variant}']

    immune = pd.concat([
        v.rename(f'immune_{k}') for k, v in immune.items()
    ], axis=1)
    return immune


def _make_vaccinations(compartments: pd.DataFrame) -> pd.DataFrame:
    vaccinations = defaultdict(lambda: pd.Series(0., index=compartments.index))

    for risk_group in RISK_GROUP_NAMES:
        for measure in ['Vaccination', 'Booster']:
            key = f'{measure}_all_all_{risk_group}'
            vaccinations[f'{measure.lower()}s_{risk_group}'] += compartments[key]
            vaccinations[f'{measure.lower()}s'] += compartments[key]

    vaccinations = pd.concat([
        v.rename(k) for k, v in vaccinations.items()
    ], axis=1)
    return vaccinations


def _make_force_of_infection(infections: pd.DataFrame,
                             susceptible: pd.DataFrame) -> pd.DataFrame:
    # Drop the first entry 'none'
    variant_names = VARIANT_NAMES[1:]
    foi = pd.DataFrame(index=infections.index)

    for key in list(variant_names) + ['total']:
        foi[f'force_of_infection_{key}'] = (
            infections[f'modeled_infections_{key}'] / susceptible[f'susceptible_{key}']
        )

    return foi


def _make_variant_prevalence(infections: pd.DataFrame) -> pd.DataFrame:
    # Drop the first entry 'none'
    variant_names = VARIANT_NAMES[1:]
    prevalence = pd.DataFrame(index=infections.index)

    for variant in variant_names:
        prevalence[f'variant_{variant}_prevalence'] = (
            infections[f'modeled_infections_{variant}'] / infections[f'modeled_infections_total']
        )

    return prevalence


def compute_corrected_hospital_usage(admissions: pd.Series,
                                     hospital_parameters,
                                     postprocessing_parameters: PostprocessingParameters):
    hfr = postprocessing_parameters.ihr / postprocessing_parameters.ifr
    hfr[hfr < 1] = 1.0
    hospital_usage = compute_hospital_usage(
        admissions,
        hfr,
        hospital_parameters,
    )
    corrected_hospital_census = (hospital_usage.hospital_census
                                 * postprocessing_parameters.hospital_census).fillna(method='ffill')
    corrected_icu_census = (corrected_hospital_census
                            * postprocessing_parameters.icu_census).fillna(method='ffill')

    hospital_usage.hospital_census = corrected_hospital_census
    hospital_usage.icu_census = corrected_icu_census

    return hospital_usage


def compute_r(model_params: Parameters,
              system_metrics: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame(index=system_metrics.index)

    base_params = model_params.base_parameters
    population = system_metrics.total_population
    sigma, gamma = base_params.sigma_all_infection, base_params.gamma_all_infection
    average_generation_time = int(round((1 / sigma + 1 / gamma).mean()))

    for label in list(VARIANT_NAMES[1:]) + ['total']:
        infections = system_metrics[f'modeled_infections_{label}']
        susceptible = system_metrics[f'susceptible_{label}']
        r[f'r_effective_{label}'] = (infections
                                     .groupby('location_id')
                                     .apply(lambda x: x / x.shift(average_generation_time)))
        r[f'r_controlled_{label}'] = r[f'r_effective_{label}'] * population / susceptible

    return r

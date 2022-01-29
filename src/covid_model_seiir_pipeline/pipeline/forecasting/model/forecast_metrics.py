from collections import defaultdict
import itertools

import numpy as np
import pandas as pd


from covid_model_seiir_pipeline.lib.ode_mk2.containers import Parameters
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    VACCINE_STATUS_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENT_NAMES,
    COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    compute_hospital_usage,
)


def compute_output_metrics(indices: Indices,
                           compartments: pd.DataFrame,
                           model_parameters: Parameters,
                           ode_params: pd.DataFrame,
                           hospital_parameters,
                           hospital_cf) -> pd.DataFrame:
    total_pop = (compartments
                 .loc[:, [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], COMPARTMENTS_NAMES)]]
                 .sum(axis=1)
                 .rename('total_population'))
    compartments_diff = compartments.groupby('location_id').diff()

    infections = _make_measure(compartments_diff, 'Infection', lag=0)
    deaths = _make_measure(compartments_diff, 'Death', lag=ode_params['exposure_to_death'].iloc[0])
    admissions = _make_measure(compartments_diff, 'Admission', lag=ode_params['exposure_to_admission'].iloc[0])
    cases = _make_measure(compartments_diff, 'Case', lag=ode_params['exposure_to_case'].iloc[0])
    susceptible = _make_susceptible(compartments_diff)
    immune = _make_immune(susceptible, total_pop)
    infectious = _make_infectious(compartments)
    vaccinations = _make_vaccinations(compartments)
    betas = compartments_diff.filter(like='Beta_none_none_all').mean(axis=1).rename('beta')
    force_of_infection = _make_force_of_infection(infections, susceptible)
    covid_status = _make_covid_status(compartments)
    variant_prevalence = _make_variant_prevalence(infections)

    hospital_usage = compute_corrected_hospital_usage(
        admissions,
        deaths,
        ode_params,
        hospital_parameters,
        hospital_cf,
    )

    system_metrics = pd.concat([
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
        covid_status,
        variant_prevalence,
        total_pop,
        hospital_usage,
        pd.concat(hospital_cf.to_dict().values(), axis=1).rename(columns=lambda x: f'{x}_correction_factor'),
    ], axis=1)

    r = compute_r(model_parameters, system_metrics)
    system_metrics = pd.concat([system_metrics, r], axis=1)

    return system_metrics


def _make_measure(compartments_diff: pd.DataFrame, measure: str, lag: int) -> pd.DataFrame:
    # Ignore 'none'
    variant_names = VARIANT_NAMES[1:]
    compartments_diff = compartments_diff.groupby('location_id').shift(lag)
    data = defaultdict(lambda: pd.Series(0., index=compartments_diff.index))

    data['naive_unvaccinated'] = compartments_diff.filter(like=f'{measure}_none_all_unvaccinated').sum(axis=1)
    data['unvaccinated'] = compartments_diff.filter(like=f'{measure}_all_all_unvaccinated').sum(axis=1)
    data['vaccinated'] = compartments_diff.filter(like=f'{measure}_none_all_vaccinated').sum(axis=1)
    data['booster'] = compartments_diff.filter(like=f'{measure}_none_all_booster').sum(axis=1)
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
        for measure, group in [('Vaccination', 'unvaccinated'), ('Booster', 'vaccinated')]:
            key = f'{measure}_all_all_{group}_{risk_group}'
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


def _make_covid_status(compartments: pd.DataFrame) -> pd.DataFrame:
    covid_status = defaultdict(lambda: pd.Series(0., index=compartments.index))

    groups = itertools.product(COMPARTMENT_NAMES, VACCINE_STATUS_NAMES, VARIANT_NAMES, RISK_GROUP_NAMES)
    for compartment, vaccine_status, variant, risk_group in groups:
        compartment_key = f'{compartment}_{vaccine_status}_{variant}_{risk_group}'
        if compartment == COMPARTMENT_NAMES.S and variant == VARIANT_NAMES.none:
            if vaccine_status == VACCINE_STATUS_NAMES.unvaccinated:
                covid_status['covid_status_naive_unvaccinated'] += compartments[compartment_key]
            else:
                covid_status['covid_status_naive_vaccinated'] += compartments[compartment_key]
        else:
            if vaccine_status == VACCINE_STATUS_NAMES.unvaccinated:
                covid_status['covid_status_exposed_unvaccinated'] += compartments[compartment_key]
            else:
                covid_status['covid_status_exposed_vaccinated'] += compartments[compartment_key]

    covid_status = pd.concat([
        v.rename(k) for k, v in covid_status.items()
    ], axis=1)
    return covid_status


def _make_variant_prevalence(infections: pd.DataFrame) -> pd.DataFrame:
    # Drop the first entry 'none'
    variant_names = VARIANT_NAMES[1:]
    prevalence = pd.DataFrame(index=infections.index)

    for variant in variant_names:
        prevalence[f'variant_{variant}_prevalence'] = (
            infections[f'modeled_infections_{variant}'] / infections[f'modeled_infections_total']
        )

    return prevalence


def compute_corrected_hospital_usage(admissions: pd.DataFrame,
                                     deaths: pd.DataFrame,
                                     ode_params: pd.DataFrame,
                                     hospital_parameters,
                                     correction_factors):
    lag = ode_params['exposure_to_death'].iloc[0] - ode_params['exposure_to_admission'].iloc[0]
    admissions = admissions['modeled_admissions_total']
    deaths = deaths['modeled_deaths_total']
    hfr = admissions.groupby('location_id').shift(lag) / deaths
    hfr[(hfr < 1) | ~np.isfinite(hfr)] = 1
    hospital_usage = compute_hospital_usage(
        admissions,
        hfr,
        hospital_parameters,
    )
    corrected_hospital_census = (hospital_usage.hospital_census
                                 * correction_factors.hospital_census).fillna(method='ffill')
    corrected_icu_census = (corrected_hospital_census
                            * correction_factors.icu_census).fillna(method='ffill')

    hospital_usage = pd.concat([
        corrected_hospital_census.rename('hospital_census'),
        hospital_usage.icu_admissions.rename('icu_admissions'),
        corrected_icu_census.rename('icu_census'),
    ], axis=1)

    return hospital_usage


def compute_r(model_params: Parameters,
              system_metrics: pd.DataFrame) -> pd.DataFrame:
    r = pd.DataFrame(index=system_metrics.index)

    base_params = model_params.base_parameters
    population = system_metrics.total_population

    base_params['sigma_total_infection'] = 0
    base_params['gamma_total_infection'] = 0
    for v in VARIANT_NAMES[1:]:
        base_params['sigma_total_infection'] += base_params[f'rho_{v}_infection'] * base_params[f'sigma_{v}_infection']
        base_params['gamma_total_infection'] += base_params[f'rho_{v}_infection'] * base_params[f'gamma_{v}_infection']

    for label in list(VARIANT_NAMES[1:]) + ['total']:
        sigma, gamma = base_params.loc[:, f'sigma_{label}_infection'], base_params.loc[:, f'gamma_{label}_infection']
        generation_time = ((1 / sigma + 1 / gamma)
                           .round()
                           .groupby('location_id').bfill()
                           .astype(int)
                           .rename('generation_time'))
        infections = system_metrics.loc[:, f'modeled_infections_{label}']
        for gt in generation_time.unique():
            r.loc[generation_time == gt, f'r_effective_{label}'] = (infections
                                                                    .groupby('location_id')
                                                                    .apply(lambda x: x / x.shift(gt)))
        susceptible = system_metrics.loc[:, f'susceptible_{label}']
        r[f'r_controlled_{label}'] = r[f'r_effective_{label}'] * population / susceptible

    return r

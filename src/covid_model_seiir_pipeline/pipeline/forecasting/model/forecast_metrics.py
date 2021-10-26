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
    HospitalMetrics,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    compute_hospital_usage,
)

if TYPE_CHECKING:
    # Support type checking but keep the pipeline stages as isolated as possible.
    from covid_model_seiir_pipeline.pipeline.regression.specification import (
        HospitalParameters,
    )


def compute_output_metrics(indices: Indices,
                           compartments: pd.DataFrame,
                           postprocessing_params: PostprocessingParameters,
                           model_parameters: Parameters,
                           hospital_parameters: 'HospitalParameters') -> Tuple[pd.DataFrame,
                                                                               pd.DataFrame]:
    system_metrics = compute_system_metrics(
        compartments,
        postprocessing_params,
    )

    infections = postprocessing_params.past_infections.loc[indices.past].append(
        system_metrics.modeled_infections_total.loc[indices.future]
    ).rename('infections')
    past_deaths = postprocessing_params.past_deaths
    modeled_deaths = system_metrics.modeled_deaths_total
    deaths = (past_deaths
              .append(modeled_deaths
                      .loc[modeled_deaths
                           .dropna()
                           .index
                           .difference(past_deaths.index)])
              .rename('deaths')
              .to_frame())
    deaths['observed'] = 0
    deaths.loc[past_deaths.index, 'observed'] = 1
    cases = (infections
             .groupby('location_id')
             .shift(postprocessing_params.infection_to_case)
             * postprocessing_params.idr).rename('cases')
    admissions = (infections
                  .groupby('location_id')
                  .shift(postprocessing_params.infection_to_admission)
                  * postprocessing_params.ihr).rename('admissions')
    hospital_usage = compute_corrected_hospital_usage(
        admissions,
        hospital_parameters,
        postprocessing_params,
    )

    r = compute_r(
        model_parameters,
        system_metrics,
    )

    output_metrics = pd.concat([
        infections,
        deaths,
        cases,
        hospital_usage.to_df(),
        r
    ], axis=1).set_index('observed', append=True)
    return system_metrics, output_metrics



def compute_system_metrics(compartments: pd.DataFrame,
                           postprocessing_params: PostprocessingParameters) -> pd.DataFrame:
    total_pop = (compartments
                 .loc[:, [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], COMPARTMENTS_NAMES)]]
                 .sum(axis=1)
                 .rename('total_population'))
    compartments_diff = compartments.groupby('location_id').diff()

    infections = _make_infections(compartments_diff)
    deaths = _make_deaths(infections, postprocessing_params)
    susceptible = _make_susceptible(compartments_diff)
    immune = _make_immune(susceptible, total_pop)
    infectious = _make_infectious(compartments)
    vaccinations = _make_vaccinations(compartments)
    betas = compartments_diff.filter(like='beta_none_all').mean(axis=1).rename('beta')
    force_of_infection = _make_force_of_infection(infections, susceptible)
    variant_prevalence = _make_variant_prevalence(infections)

    return pd.concat([
        infections,
        deaths,
        susceptible,
        immune,
        infectious,
        vaccinations,
        betas,
        force_of_infection,
        variant_prevalence,
        total_pop,
    ], axis=1)


def _make_infections(compartments_diff: pd.DataFrame) -> pd.DataFrame:
    # Ignore 'none'
    variant_names = VARIANT_NAMES[1:]
    infections = defaultdict(lambda: pd.Series(0., index=compartments_diff.index))

    for variant in variant_names:
        for vaccine_status in VACCINE_STATUS_NAMES:
            for risk_group in RISK_GROUP_NAMES:
                key = f'NewE_{variant}_{vaccine_status}_{risk_group}'
                infections[variant] += compartments_diff[key]
                infections[vaccine_status] += compartments_diff[key]
                infections[risk_group] += compartments_diff[key]
                infections['total'] += compartments_diff[key]

    infections = pd.concat([
        v.rename(f'modeled_infections_{k}') for k, v in infections.items()
    ], axis=1)

    return infections


def _make_susceptible(compartments_diff: pd.DataFrame) -> pd.DataFrame:
    variant_names = VARIANT_NAMES[1:]
    susceptible = defaultdict(lambda: pd.Series(0., index=compartments_diff.index))

    for variant in variant_names:
        for vaccine_status in VACCINE_STATUS_NAMES:
            for risk_group in RISK_GROUP_NAMES:
                key = f'EffectiveSusceptible_{variant}_{vaccine_status}_{risk_group}'
                susceptible[variant] += compartments_diff[key]
                susceptible[vaccine_status] = np.maximum(susceptible[vaccine_status], compartments_diff[key])
                susceptible[risk_group] = np.maximum(susceptible[vaccine_status], compartments_diff[key])

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
                key = f'I_{variant}_{vaccine_status}_{risk_group}'
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

    for variant in VARIANT_NAMES:
        for vaccine_status in VACCINE_STATUS_NAMES:
            for risk_group in RISK_GROUP_NAMES:
                for measure in ['Vaccination', 'Booster']:
                    key = f'New{measure}_{variant}_{vaccine_status}_{risk_group}'
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

    for key in list(variant_names) + list(VACCINE_STATUS_NAMES) + ['total']:
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


def _make_deaths(infections: pd.DataFrame,
                 postprocessing_params: PostprocessingParameters) -> pd.DataFrame:
    deaths = pd.DataFrame(index=infections.index)
    for risk_group in RISK_GROUP_NAMES:
        group_ifr = getattr(postprocessing_params, f'ifr_{risk_group}').rename('ifr')
        deaths[f'modeled_deaths_{risk_group}'] = _compute_deaths(
            infections[f'modeled_infections_{risk_group}'],
            postprocessing_params.infection_to_death,
            group_ifr,
        )
    deaths['modeled_deaths_total'] = deaths.sum(axis=1)

    return deaths


def _compute_deaths(modeled_infections: pd.Series,
                    infection_death_lag: int,
                    ifr: pd.Series) -> pd.Series:
    modeled_deaths = (modeled_infections
                      .groupby('location_id')
                      .shift(infection_death_lag) * ifr)
    modeled_deaths = modeled_deaths.rename('deaths').reset_index()
    modeled_deaths = modeled_deaths.set_index(['location_id', 'date']).deaths
    return modeled_deaths


def compute_corrected_hospital_usage(admissions: pd.Series,
                                     hospital_parameters: 'HospitalParameters',
                                     postprocessing_parameters: PostprocessingParameters) -> HospitalMetrics:
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

    population = system_metrics.total_population
    sigma, gamma = model_params.sigma_all, model_params.gamma_all
    average_generation_time = int(round((1 / sigma + 1 / gamma).mean()))

    for label in list(VARIANT_NAMES[1:]) + ['total']:
        infections = system_metrics[f'modeled_infections_{label}']
        susceptible = system_metrics[f'susceptible_{label}']
        r[f'r_effective_{label}'] = (infections
                                     .groupby('location_id')
                                     .apply(lambda x: x / x.shift(average_generation_time)))
        r[f'r_controlled_{label}'] = r[f'r_effective_{label}'] * population / susceptible

    return r

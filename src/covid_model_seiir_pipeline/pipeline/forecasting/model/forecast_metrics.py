import itertools
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd


from covid_model_seiir_pipeline.lib.ode_mk2.containers import Parameters
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT,
    VARIANT_NAMES,
    VACCINE_STATUS_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    PostprocessingParameters,
    HospitalMetrics,
    SystemMetrics,
    OutputMetrics,
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
                           hospital_parameters: 'HospitalParameters') -> Tuple[SystemMetrics,
                                                                               OutputMetrics]:
    system_metrics = variant_system_metrics(
        model_parameters,
        postprocessing_params,
        compartments,
    )

    infections = postprocessing_params.past_infections.loc[indices.past].append(
        system_metrics.modeled_infections_total.loc[indices.future]
    )
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
    deaths = deaths.set_index('observed', append=True).deaths
    cases = (infections
             .groupby('location_id')
             .shift(postprocessing_params.infection_to_case)
             * postprocessing_params.idr)
    admissions = (infections
                  .groupby('location_id')
                  .shift(postprocessing_params.infection_to_admission)
                  * postprocessing_params.ihr)
    hospital_usage = compute_corrected_hospital_usage(
        admissions,
        hospital_parameters,
        postprocessing_params,
    )

    effective_r = compute_effective_r(
        model_parameters,
        system_metrics,
    )

    output_metrics = OutputMetrics(
        infections=infections,
        cases=cases,
        deaths=deaths,
        **hospital_usage.to_dict(),
        # Other stuff
        **effective_r,
    )
    return system_metrics, output_metrics


def variant_system_metrics(model_parameters: Parameters,
                           postprocessing_params: PostprocessingParameters,
                           components: pd.DataFrame) -> SystemMetrics:
    components_diff = components.groupby('location_id').diff()

    infections = _make_infections(components_diff)
    infected = infections['modeled_infections_total'] - infections['modeled_infections_natural_breakthrough']
    group_infections, group_deaths = _make_group_infections_and_deaths_metrics(
        components,
        components_diff,
        postprocessing_params,
    )

    susceptible = _make_susceptible(components)
    infectious = _make_infectious(components)
    immune = _make_immune(components)
    vaccinations = _make_vaccinations(components, components_diff)
    total_pop = components[[f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], COMPARTMENTS_NAMES)]].sum(axis=1)
    betas = _make_betas(components)
    incidence = _make_incidence(infections, total_pop)
    force_of_infection = _make_force_of_infection(infections, susceptible)

    variant_prevalence = infections['modeled_infections_variant'] / infections['modeled_infections_total']
    proportion_cross_immune = pd.Series(np.nan, index=components.index)

    return SystemMetrics(
        **infections,
        modeled_infected_total=infected,
        **group_infections,
        **group_deaths,
        **susceptible,
        **infectious,
        **immune,
        **vaccinations,
        total_population=total_pop,
        **betas,
        **incidence,
        **force_of_infection,
        variant_prevalence=variant_prevalence,
        proportion_cross_immune=proportion_cross_immune,
    )


def _make_infections(components_diff) -> Dict[str, pd.Series]:
    output_column_map = {
        'wild': ('NewE_ancestral_unvaccinated',
                 'NewE_ancestral_vaccinated',
                 'NewE_ancestral_booster',
                 'NewE_alpha_unvaccinated',
                 'NewE_alpha_vaccinated',
                 'NewE_alpha_booster',),
        'variant': ('NewE_beta_unvaccinated',
                    'NewE_beta_vaccinated',
                    'NewE_beta_booster',
                    'NewE_gamma_unvaccinated',
                    'NewE_gamma_vaccinated',
                    'NewE_gamma_booster',
                    'NewE_delta_unvaccinated',
                    'NewE_delta_vaccinated',
                    'NewE_delta_booster',
                    'NewE_other_unvaccinated',
                    'NewE_other_vaccinated',
                    'NewE_other_booster',
                    'NewE_omega_unvaccinated',
                    'NewE_omega_vaccinated',
                    'NewE_omega_booster',),
        'natural_breakthrough': (),
        'vaccine_breakthrough': ('NewE_none_vaccinated',
                                 'NewE_ancestral_vaccinated',
                                 'NewE_alpha_vaccinated',
                                 'NewE_beta_vaccinated',
                                 'NewE_gamma_vaccinated',
                                 'NewE_delta_vaccinated',
                                 'NewE_other_vaccinated',
                                 'NewE_omega_vaccinated',),
        'total': ('NewE_none_unvaccinated',
                  'NewE_none_vaccinated',
                  'NewE_none_booster',
                  'NewE_ancestral_unvaccinated',
                  'NewE_ancestral_vaccinated',
                  'NewE_ancestral_booster',
                  'NewE_alpha_unvaccinated',
                  'NewE_alpha_vaccinated',
                  'NewE_alpha_booster',
                  'NewE_beta_unvaccinated',
                  'NewE_beta_vaccinated',
                  'NewE_beta_booster',
                  'NewE_gamma_unvaccinated',
                  'NewE_gamma_vaccinated',
                  'NewE_gamma_booster',
                  'NewE_delta_unvaccinated',
                  'NewE_delta_vaccinated',
                  'NewE_delta_booster',
                  'NewE_other_unvaccinated',
                  'NewE_other_vaccinated',
                  'NewE_other_booster',
                  'NewE_omega_unvaccinated',
                  'NewE_omega_vaccinated',
                  'NewE_omega_booster',),
        'unvaccinated_wild': ('NewE_none_unvaccinated',
                              'NewE_ancestral_unvaccinated',
                              'NewE_alpha_unvaccinated',),
        'unvaccinated_variant': ('NewE_beta_unvaccinated',
                                 'NewE_gamma_unvaccinated',
                                 'NewE_delta_unvaccinated',
                                 'NewE_other_unvaccinated',
                                 'NewE_omega_unvaccinated',),
        'unvaccinated_natural_breakthrough': (),
        'unvaccinated_total': ('NewE_none_unvaccinated',
                               'NewE_ancestral_unvaccinated',
                               'NewE_alpha_unvaccinated',
                               'NewE_beta_unvaccinated',
                               'NewE_gamma_unvaccinated',
                               'NewE_delta_unvaccinated',
                               'NewE_other_unvaccinated',
                               'NewE_omega_unvaccinated',),
    }
    return _make_outputs(components_diff, 'modeled_infections', output_column_map)


def _make_susceptible(components_diff) -> Dict[str, pd.Series]:
    output_column_map = {
        'wild': ('EffectiveSusceptible_ancestral_unvaccinated',
                 'EffectiveSusceptible_ancestral_vaccinated',
                 'EffectiveSusceptible_ancestral_booster',),
        'variant': ('EffectiveSusceptible_delta_unvaccinated',
                    'EffectiveSusceptible_delta_vaccinated',
                    'EffectiveSusceptible_delta_booster',),
        'variant_only': (),
        'variant_unprotected': (),
        'unvaccinated_wild': ('EffectiveSusceptible_ancestral_unvaccinated',),
        'unvaccinated_variant': ('EffectiveSusceptible_delta_unvaccinated',),
        'unvaccinated_variant_only': (),
    }
    return _make_outputs(components_diff, 'total_susceptible', output_column_map)


def _make_infectious(components) -> Dict[str, pd.Series]:
    output_column_map = {
        'wild': ('I_none_unvaccinated',
                 'I_none_vaccinated',
                 'I_none_booster',
                 'I_ancestral_unvaccinated',
                 'I_ancestral_vaccinated',
                 'I_ancestral_booster',
                 'I_alpha_unvaccinated',
                 'I_alpha_vaccinated',
                 'I_alpha_booster',),
        'variant': ('I_beta_unvaccinated',
                    'I_beta_vaccinated',
                    'I_beta_booster',
                    'I_gamma_unvaccinated',
                    'I_gamma_vaccinated',
                    'I_gamma_booster',
                    'I_delta_unvaccinated',
                    'I_delta_vaccinated',
                    'I_delta_booster',
                    'I_other_unvaccinated',
                    'I_other_vaccinated',
                    'I_other_booster',
                    'I_omega_unvaccinated',
                    'I_omega_vaccinated',
                    'I_omega_booster',),
        '': ('I_none_unvaccinated',
             'I_none_vaccinated',
             'I_none_booster',
             'I_ancestral_unvaccinated',
             'I_ancestral_vaccinated',
             'I_ancestral_booster',
             'I_alpha_unvaccinated',
             'I_alpha_vaccinated',
             'I_alpha_booster',
             'I_beta_unvaccinated',
             'I_beta_vaccinated',
             'I_beta_booster',
             'I_gamma_unvaccinated',
             'I_gamma_vaccinated',
             'I_gamma_booster',
             'I_delta_unvaccinated',
             'I_delta_vaccinated',
             'I_delta_booster',
             'I_other_unvaccinated',
             'I_other_vaccinated',
             'I_other_booster',
             'I_omega_unvaccinated',
             'I_omega_vaccinated',
             'I_omega_booster',),
    }
    return _make_outputs(components, 'total_infectious', output_column_map)


def _make_immune(components) -> Dict[str, pd.Series]:
    output_column_map = {
        'wild': (),
        'variant': (),
    }
    return _make_outputs(components, 'total_immune', output_column_map)


def _make_vaccinations(components, components_diff) -> Dict[str, pd.Series]:
    output_column_map = {
        'ineffective': (),
        'protected_wild': (),
        'protected_all': (),
        'immune_wild': (),
        'immune_all': (),
        'effective': tuple([c for c in TRACKING_COMPARTMENTS_NAMES if 'NewVaccination' in c]),
    }
    vaccinations = _make_outputs(components, 'vaccinations', output_column_map)
    vaccinations.update(
        _make_outputs(components, 'vaccinations', {
           'n_unvaccinated': tuple([c for c in COMPARTMENTS_NAMES if '_unvaccinated' in c])
        })
    )
    return vaccinations


def _make_betas(components):
    beta = components.filter(like='beta_none_all').mean(axis=1).groupby('location_id').diff().rename('beta')
    betas = {
        'beta': beta,
        'beta_wild': pd.Series(np.nan, index=beta.index),
        'beta_variant': pd.Series(np.nan, index=beta.index),
    }
    return betas


def _make_incidence(infections: Dict[str, pd.Series], total_pop: pd.Series) -> Dict[str, pd.Series]:
    return {
        f'incidence_{i_type}': infections[f'modeled_infections_{i_type}'] / total_pop
        for i_type in ['wild', 'variant', 'total', 'unvaccinated_wild', 'unvaccinated_variant', 'unvaccinated_total']
    }


def _make_force_of_infection(infections: Dict[str, pd.Series],
                             susceptible: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    foi = infections['modeled_infections_total'] / susceptible['total_susceptible_variant']
    foi_novax = (
        infections['modeled_infections_unvaccinated_total']
        / susceptible['total_susceptible_unvaccinated_variant']
    )
    foi_novax_naive = (
        (infections['modeled_infections_unvaccinated_total']
         - infections['modeled_infections_unvaccinated_natural_breakthrough'])
        / susceptible['total_susceptible_unvaccinated_wild'])

    foi_novax_breakthrough = (
        infections['modeled_infections_unvaccinated_natural_breakthrough']
        / susceptible['total_susceptible_unvaccinated_variant_only']
    )
    return {
        'force_of_infection': foi,
        'force_of_infection_unvaccinated': foi_novax,
        'force_of_infection_unvaccinated_naive': foi_novax_naive,
        'force_of_infection_unvaccinated_natural_breakthrough': foi_novax_breakthrough,
    }


def _make_outputs(data, prefix, column_map):
    out = {}
    for suffix, base_cols in column_map.items():
        cols = [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], base_cols)]
        key = f'{prefix}_{suffix}' if suffix else prefix
        if cols:
            out[key] = data[cols].sum(axis=1)
        else:
            out[key] = pd.Series(np.nan, index=data.index)
    return out


def _make_group_infections_and_deaths_metrics(
        components: pd.DataFrame,
        components_diff: pd.DataFrame,
        postprocessing_params: PostprocessingParameters
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    group_infections = {}
    group_deaths = {}
    for group in ['hr', 'lr']:
        group_compartments = [c for c in components.columns if group in c]
        group_components_diff = components_diff.loc[:, group_compartments]
        group_ifr = getattr(postprocessing_params, f'ifr_{group}').rename('ifr')

        for covid_type, variants in [('wild', ('ancestral', 'alpha')),
                                     ('variant', ('beta', 'gamma', 'delta', 'other', 'omega'))]:
            cols = [f'NewE_{variant}_{vaccine_status}_{group}'
                    for variant, vaccine_status in itertools.product(variants, VACCINE_STATUS_NAMES)]
            infections = group_components_diff[cols].sum(axis=1).rename('infections')
            deaths = _compute_deaths(
                infections,
                postprocessing_params.infection_to_death,
                group_ifr,
            )
            group_infections[(group, covid_type)] = infections
            group_deaths[(group, covid_type)] = deaths
    modeled_infections = dict(
        modeled_infections_lr=group_infections[('lr', 'wild')] + group_infections[('lr', 'variant')],
        modeled_infections_hr=group_infections[('hr', 'wild')] + group_infections[('hr', 'variant')],
    )

    modeled_deaths = dict(
        modeled_deaths_wild=group_deaths[('lr', 'wild')] + group_deaths[('hr', 'wild')],
        modeled_deaths_variant=group_deaths[('lr', 'variant')] + group_deaths[('hr', 'variant')],
        modeled_deaths_lr=group_deaths[('lr', 'wild')] + group_deaths[('lr', 'variant')],
        modeled_deaths_hr=group_deaths[('hr', 'wild')] + group_deaths[('hr', 'variant')],
        modeled_deaths_total=sum(group_deaths.values()),
    )
    return modeled_infections, modeled_deaths


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


def compute_effective_r(model_params: Parameters,
                        system_metrics: SystemMetrics) -> Dict[str, pd.Series]:
    sigma, gamma = model_params.sigma_all, model_params.gamma_all
    average_generation_time = int(round((1 / sigma + 1 / gamma).mean()))

    system_metrics = system_metrics.to_dict()
    pop = system_metrics['total_population']
    out = {}
    for label in ['wild', 'variant', 'total']:
        suffix = '' if label == 'total' else f'_{label}'
        s_label = 'variant' if label == 'total' else label
        infections = system_metrics[f'modeled_infections_{label}']
        susceptible = system_metrics[f'total_susceptible_{s_label}']
        out[f'r_effective{suffix}'] = (infections
                                       .groupby('location_id')
                                       .apply(lambda x: x / x.shift(average_generation_time)))
        out[f'r_controlled{suffix}'] = out[f'r_effective{suffix}'] * pop / susceptible

    return out

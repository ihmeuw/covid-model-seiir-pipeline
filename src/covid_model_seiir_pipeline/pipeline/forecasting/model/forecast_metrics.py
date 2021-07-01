import itertools
from typing import Tuple, TYPE_CHECKING

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    ode,
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
                           future_components: pd.DataFrame,
                           postprocessing_params: PostprocessingParameters,
                           model_parameters: ode.ForecastParameters,
                           hospital_parameters: 'HospitalParameters') -> Tuple[pd.DataFrame,
                                                                               SystemMetrics,
                                                                               OutputMetrics]:
    components = postprocessing_params.past_compartments
    components = (components
                  .loc[indices.past]  # Need to drop transition day.
                  .append(future_components)
                  .sort_index())

    system_metrics = variant_system_metrics(
        indices,
        model_parameters,
        postprocessing_params,
        components,
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

    rcw, rew, rcv, rev, re = compute_effective_r(
        model_parameters,
        system_metrics,
        infections,

    )

    output_metrics = OutputMetrics(
        infections=infections,
        cases=cases,
        deaths=deaths,
        **hospital_usage.to_dict(),
        # Other stuff
        r_controlled_wild=rcw,
        r_effective_wild=rew,
        r_controlled_variant=rcv,
        r_effective_variant=rev,
        r_effective=re,
    )
    return components, system_metrics, output_metrics


def variant_system_metrics(indices: Indices,
                           model_parameters: ode.ForecastParameters,
                           postprocessing_params: PostprocessingParameters,
                           components: pd.DataFrame) -> SystemMetrics:
    components_diff = components.groupby('location_id').diff()

    cols = [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], ode.COMPARTMENTS._fields)]
    tracking_cols = [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], ode.TRACKING_COMPARTMENTS._fields)]
    modeled_infections_wild = components_diff[[c for c in tracking_cols if 'NewE_wild' in c]].sum(axis=1)
    modeled_infections_variant = components_diff[[c for c in tracking_cols if 'NewE_variant' in c]].sum(axis=1)
    natural_immunity_breakthrough = components_diff[[c for c in tracking_cols if 'NewE_nbt' in c]].sum(axis=1)
    vaccine_breakthrough = components_diff[[c for c in tracking_cols if 'NewE_vbt' in c]].sum(axis=1)
    new_s_variant = components_diff[[c for c in tracking_cols if 'NewS_v' in c]].sum(axis=1)
    new_r_wild = components_diff[[c for c in tracking_cols if 'NewR_w' in c]].sum(axis=1)
    proportion_cross_immune = new_r_wild / (new_s_variant + new_r_wild)

    modeled_deaths_wild = pd.Series(0, index=indices.full)
    modeled_deaths_variant = pd.Series(0, index=indices.full)
    for group in ['hr', 'lr']:
        group_compartments = [c for c in components.columns if group in c]
        group_components_diff = components_diff.loc[:, group_compartments]
        group_ifr = getattr(postprocessing_params, f'ifr_{group}').rename('ifr')

        group_deaths = {}
        for covid_type in ['wild', 'variant']:
            group_infections = group_components_diff[f'NewE_{covid_type}_{group}'].rename('infections')
            group_infections_p = group_components_diff[f'NewE_p_{covid_type}_{group}'].rename('infections')
            group_infections_not_p = group_infections - group_infections_p
            group_deaths[covid_type] = compute_deaths(
                group_infections_not_p,
                postprocessing_params.infection_to_death,
                group_ifr,
            )
        modeled_deaths_wild += group_deaths['wild']
        modeled_deaths_variant += group_deaths['variant']

    vaccines_u = components_diff[[c for c in tracking_cols if 'V_u' in c]].sum(axis=1)
    vaccines_p = components_diff[[c for c in tracking_cols if 'V_p' in c and 'pa' not in c]].sum(axis=1)
    vaccines_pa = components_diff[[c for c in tracking_cols if 'V_pa' in c]].sum(axis=1)
    vaccines_m = components_diff[[c for c in tracking_cols if 'V_m' in c and 'ma' not in c]].sum(axis=1)
    vaccines_ma = components_diff[[c for c in tracking_cols if 'V_ma' in c]].sum(axis=1)

    s_wild = components[[c for c in cols if 'S' in c and 'variant' not in c and 'm' not in c]].sum(axis=1)
    s_variant = components[[c for c in cols if 'S' in c]].sum(axis=1)
    s_variant_only = components[[c for c in cols if 'S_variant' in c or 'S_m' in c]].sum(axis=1)
    s_variant_unprotected = components[[c for c in cols if 'S' in c and 'pa' not in c and 'm' not in c]].sum(axis=1)

    i_wild = components[[c for c in cols if 'I' in c and 'variant' not in c]].sum(axis=1)
    i_variant = components[[c for c in cols if 'I' in c and 'variant' in c]].sum(axis=1)
    i_total = i_wild + i_variant

    total_population = components[cols].sum(axis=1)

    total_pop = components[cols].sum(axis=1)
    beta_wild = modeled_infections_wild / (s_wild * i_wild ** model_parameters.alpha / total_pop)
    beta_variant = modeled_infections_variant / (s_variant * i_variant ** model_parameters.alpha / total_pop)
    modeled_infections_total = modeled_infections_wild + modeled_infections_variant
    beta_total = modeled_infections_total / (s_variant * i_total ** model_parameters.alpha / total_pop)

    immune_variant = components[[c for c in cols if 'R' in c]].sum(axis=1)
    immune_wild = immune_variant + s_variant_only

    return SystemMetrics(
        modeled_infections_wild=modeled_infections_wild,
        modeled_infections_variant=modeled_infections_variant,
        modeled_infections_total=modeled_infections_wild + modeled_infections_variant,
        modeled_infected_total=modeled_infections_wild + modeled_infections_variant - natural_immunity_breakthrough,

        variant_prevalence=modeled_infections_variant / (modeled_infections_wild + modeled_infections_variant),

        natural_immunity_breakthrough=natural_immunity_breakthrough,
        vaccine_breakthrough=vaccine_breakthrough,
        proportion_cross_immune=proportion_cross_immune,

        modeled_deaths_wild=modeled_deaths_wild,
        modeled_deaths_variant=modeled_deaths_variant,
        modeled_deaths_total=modeled_deaths_wild + modeled_deaths_variant,

        vaccinations_protected_wild=vaccines_p,
        vaccinations_protected_all=vaccines_pa,
        vaccinations_immune_wild=vaccines_m,
        vaccinations_immune_all=vaccines_ma,
        vaccinations_effective=vaccines_p + vaccines_pa + vaccines_m + vaccines_ma,
        vaccinations_ineffective=vaccines_u,

        total_susceptible_wild=s_wild,
        total_susceptible_variant=s_variant,
        total_susceptible_variant_only=s_variant_only,
        total_susceptible_variant_unprotected=s_variant_unprotected,
        total_infectious_wild=i_wild,
        total_infectious_variant=i_variant,
        total_immune_wild=immune_wild,
        total_immune_variant=immune_variant,

        total_population=total_population,

        beta=beta_total,
        beta_wild=beta_wild,
        beta_variant=beta_variant,
    )


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


def compute_deaths(modeled_infections: pd.Series,
                   infection_death_lag: int,
                   ifr: pd.Series) -> pd.Series:
    modeled_deaths = (modeled_infections
                      .groupby('location_id')
                      .shift(infection_death_lag) * ifr)
    modeled_deaths = modeled_deaths.rename('deaths').reset_index()
    modeled_deaths['observed'] = 0
    modeled_deaths = modeled_deaths.set_index(['location_id', 'date', 'observed']).deaths
    return modeled_deaths


def compute_effective_r(model_params: ode.ForecastParameters,
                        system_metrics: SystemMetrics,
                        infections: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    alpha, sigma = model_params.alpha, model_params.sigma
    beta_wild, beta_variant = model_params.beta_wild, model_params.beta_variant
    gamma1, gamma2 = model_params.gamma1, model_params.gamma2

    s_wild, s_variant = system_metrics.total_susceptible_wild, system_metrics.total_susceptible_variant
    i_wild, i_variant = system_metrics.total_infectious_wild, system_metrics.total_infectious_variant
    population = system_metrics.total_population

    avg_gamma_wild = 1 / (1 / gamma1 + 1 / gamma2)
    r_controlled_wild = (
        beta_wild * alpha * sigma / avg_gamma_wild * i_wild**(alpha - 1)
    ).rename('r_controlled_wild')
    r_effective_wild = (r_controlled_wild * s_wild / population).rename('r_effective_wild')

    avg_gamma_variant = 1 / (1 / gamma1 + 1 / gamma2)
    r_controlled_variant = (
        beta_variant * alpha * sigma / avg_gamma_variant * i_variant**(alpha - 1)
    ).rename('r_controlled_variant')
    r_effective_variant = (r_controlled_variant * s_variant / population).rename('r_effective_variant')

    average_generation_time = int(round((1 / sigma + 1 / gamma1 + 1 / gamma2).mean()))
    r_effective_empirical = infections.groupby('location_id').apply(lambda x: x / x.shift(average_generation_time))

    return (
        r_controlled_wild,
        r_effective_wild,
        r_controlled_variant,
        r_effective_variant,
        r_effective_empirical,
    )

import itertools
from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    ModelParameters,
    PostprocessingParameters,
    HospitalMetrics,
    SystemMetrics,
    OutputMetrics,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    compute_hospital_usage,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.ode_systems import (
    variant,
)


if TYPE_CHECKING:
    # Support type checking but keep the pipeline stages as isolated as possible.
    from covid_model_seiir_pipeline.pipeline.regression.specification import (
        HospitalParameters,
    )


def compute_output_metrics(indices: Indices,
                           future_components: pd.DataFrame,
                           postprocessing_params: PostprocessingParameters,
                           model_parameters: ModelParameters,
                           hospital_parameters: 'HospitalParameters',
                           system: str) -> Tuple[pd.DataFrame, SystemMetrics, OutputMetrics]:
    components = postprocessing_params.past_compartments
    components = (components
                  .loc[indices.past]  # Need to drop transition day.
                  .append(future_components)
                  .sort_index())

    system_metrics = build_system_metrics(
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

    output_metrics = OutputMetrics(
        infections=infections,
        cases=cases,
        deaths=deaths,
        **hospital_usage.to_dict(),
        # Other stuff
        r_controlled=pd.Series(np.nan, index=indices.full),
        r_effective=pd.Series(np.nan, index=indices.full),
        herd_immunity=pd.Series(np.nan, index=indices.full),
    )
    return components, system_metrics, output_metrics


def build_system_metrics(indices: Indices,
                         model_parameters: ModelParameters,
                         postprocessing_params: PostprocessingParameters,
                         components: pd.DataFrame) -> SystemMetrics:
    components_diff = components.groupby('location_id').diff()
    cols = components_diff.columns

    modeled_infections = pd.Series(0, index=indices.full, name='infections')
    modeled_deaths = pd.Series(0, index=indices.full, name='deaths')
    effective_vaccinations = pd.Series(0, index=indices.full, name='effective_vaccinations')
    for group in ['hr', 'lr']:
        group_compartments = [c for c in cols if group in c]
        group_components_diff = components_diff.loc[:, group_compartments]

        delta_s = (group_components_diff
                   .loc[:, [c for c in group_compartments if 'S' in c and 'S_m' not in c]]
                   .sum(axis=1))
        delta_p = (group_components_diff
                   .loc[:, [c for c in group_compartments if '_p' in c]]
                   .sum(axis=1))
        delta_new_e_p = (group_components_diff
                         .loc[:, [c for c in group_compartments if '_p' in c and 'S' not in c]]
                         .sum(axis=1))
        delta_r_m = (group_components_diff
                     .loc[:, [c for c in group_compartments if '_m' in c]]
                     .sum(axis=1))

        group_modeled_infections = -(delta_s + delta_r_m).rename('infections')
        group_vulnerable_infections = -(delta_s + delta_new_e_p + delta_r_m).rename('infections')
        group_ifr = getattr(postprocessing_params, f'ifr_{group}').rename('ifr')
        group_deaths = compute_deaths(
            group_vulnerable_infections,
            postprocessing_params.infection_to_death,
            group_ifr
        )
        group_effective_vaccinations = (delta_p + delta_r_m).rename('effective_vaccinations')

        modeled_infections += group_modeled_infections
        modeled_deaths += group_deaths
        effective_vaccinations += group_effective_vaccinations

    s = components[[c for c in cols if 'S' in c and '_m' not in c]].sum(axis=1)
    i = components[[c for c in cols if 'I' in c]].sum(axis=1)
    total_pop = components.sum(axis=1)
    beta = modeled_infections / (s * i ** model_parameters.alpha / total_pop)

    total_immune = components[[c for c in components.columns if 'R' in c]].sum(axis=1)

    return SystemMetrics(
        modeled_infections_wild=modeled_infections,
        modeled_infections_variant=pd.Series(np.nan, index=indices.full),
        modeled_infections_total=modeled_infections,

        variant_prevalence=pd.Series(np.nan, index=indices.full),
        natural_immunity_breakthrough=pd.Series(np.nan, index=indices.full),
        vaccine_breakthrough=pd.Series(np.nan, index=indices.full),
        proportion_cross_immune=pd.Series(np.nan, index=indices.full),

        modeled_deaths_wild=modeled_deaths,
        modeled_deaths_variant=pd.Series(np.nan, index=indices.full),
        modeled_deaths_total=modeled_deaths,

        vaccinations_protected_wild=pd.Series(np.nan, index=indices.full),
        vaccinations_protected_all=pd.Series(np.nan, index=indices.full),
        vaccinations_immune_wild=pd.Series(np.nan, index=indices.full),
        vaccinations_immune_all=pd.Series(np.nan, index=indices.full),
        vaccinations_effective=effective_vaccinations,
        vaccinations_ineffective=pd.Series(np.nan, index=indices.full),

        total_susceptible_wild=s,
        total_susceptible_variant=pd.Series(np.nan, index=indices.full),
        total_immune_wild=total_immune,
        total_immune_variant=pd.Series(np.nan, index=indices.full),

        beta=beta,
        beta_wild=pd.Series(np.nan, index=indices.full),
        beta_variant=pd.Series(np.nan, index=indices.full),
    )


def variant_system_metrics(indices: Indices,
                           model_parameters: ModelParameters,
                           postprocessing_params: PostprocessingParameters,
                           components: pd.DataFrame) -> SystemMetrics:
    components_diff = components.groupby('location_id').diff()
    cols = [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], variant.REAL_COMPARTMENTS)]
    tracking_cols = [f'{c}_{g}' for g, c in itertools.product(['lr', 'hr'], variant.TRACKING_COMPARTMENTS)]
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

    i_wild = components[[c for c in cols if 'I' in c and 'variant' not in c]].sum(axis=1)
    i_variant = components[[c for c in cols if 'I' in c and 'variant' in c]].sum(axis=1)
    i_total = i_wild + i_variant

    total_pop = components[cols].sum(axis=1)
    beta_wild = modeled_infections_wild / (s_wild * i_wild ** model_parameters.alpha / total_pop)
    beta_variant = modeled_infections_variant / (s_variant * i_variant ** model_parameters.alpha / total_pop)
    modeled_infections_total = modeled_infections_wild + modeled_infections_variant
    beta_total = modeled_infections_total / (s_variant * i_total ** model_parameters.alpha / total_pop)

    immune_wild = components[[c for c in cols if ('R' in c or 'S_m' in c) and 'variant' not in c]].sum(axis=1)
    immune_variant = components[[c for c in cols if 'R' in c]].sum(axis=1)

    return SystemMetrics(
        modeled_infections_wild=modeled_infections_wild,
        modeled_infections_variant=modeled_infections_variant,
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
        total_immune_wild=immune_wild,
        total_immune_variant=immune_variant,

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
    hospital_usage.hospital_census = (hospital_usage.hospital_census
                                      * postprocessing_parameters.hospital_census).fillna(method='ffill')
    hospital_usage.icu_census = (hospital_usage.icu_census
                                 * postprocessing_parameters.icu_census).fillna(method='ffill')
    hospital_usage.ventilator_census = (hospital_usage.ventilator_census
                                        * postprocessing_parameters.ventilator_census).fillna(method='ffill')
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


def compute_effective_r(components: pd.DataFrame,
                        model_params: ModelParameters) -> Tuple[pd.Series, pd.Series]:
    alpha, sigma = model_params.alpha, model_params.sigma
    gamma1, gamma2 = model_params.gamma1, model_params.gamma2

    components = components.reset_index().set_index(['location_id', 'date'])

    beta, theta = model_params.beta, model_params.theta_minus
    susceptible = components[[c for c in components.columns if 'S' in c]].sum(axis=1)
    infected = components[[c for c in components.columns if 'I' in c]].sum(axis=1)
    n = components[[c for c in components.columns if 'beta' not in c]].sum(axis=1).groupby('location_id').max()
    avg_gamma = 1 / (1 / (gamma1*(sigma - theta)) + 1 / (gamma2*(sigma - theta)))

    r_controlled = (beta * alpha * sigma / avg_gamma * infected**(alpha - 1)).rename('r_controlled')
    r_effective = (r_controlled * susceptible / n).rename('r_effective')

    return r_controlled, r_effective

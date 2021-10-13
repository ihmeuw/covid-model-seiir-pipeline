import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT,
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    solver,
)


def prepare_ode_fit_parameters(past_infections: pd.Series,
                               rhos: pd.DataFrame,
                               vaccinations: pd.DataFrame,
                               boosters: pd.DataFrame,
                               etas_vaccination: pd.DataFrame,
                               etas_booster: pd.DataFrame,
                               phis: pd.DataFrame,
                               sampled_params: Dict[str, pd.Series]) -> Parameters:
    past_index = past_infections.index
    sampled_params = {k if 'kappa' in k else f'{k}_all': v for k, v in sampled_params.items()}
    
    beta = pd.Series(-1, index=past_index, name='beta_all')
    rhos = rhos.reindex(past_index, fill_value=0.).to_dict('series')
    rhos['rho_none'] = pd.Series(0., index=past_index, name='rho_none')
    
    vaccinations = vaccinations.reindex(past_index, fill_value=0.).to_dict('series')
    boosters = (boosters
                .reindex(past_index, fill_value=0.)
                .rename(columns=lambda x: x.replace('vaccinations', 'boosters'))
                .to_dict('series'))
    
    etas_unvaccinated = etas_vaccination.copy()
    etas_unvaccinated.loc[:, :] = 0.
    etas = {}
    for prefix, eta in (('unvaccinated', etas_unvaccinated), ('vaccinated', etas_vaccination), ('booster', etas_booster)):
        eta = eta.reindex(past_index).groupby('location_id').bfill().fillna(0.).rename(columns=lambda x: f'eta_{prefix}_{x}')
        etas.update(eta.to_dict('series'))
    
    betas = {f'beta_': pd.Series(-1, index=past_index, name=f'beta_{variant}') for variant in VARIANT_NAMES}

    return Parameters(
        **sampled_params,
        new_e_all=past_infections,
        beta_all=beta,        
        **rhos,
        **vaccinations,
        **boosters,
        **etas,
        **phis.to_dict('series'),
    )


def sample_params(past_index: pd.Index,
                  param_dict: Dict,
                  params_to_sample: List[str]) -> Dict[str, pd.Series]:
    sampled_params = {}
    for parameter in params_to_sample:
        param_spec = param_dict[parameter]
        if isinstance(param_spec, (int, float)):
            value = param_spec
        else:
            value = np.random.uniform(*param_spec)

        sampled_params[parameter] = pd.Series(
            value,
            index=past_index,
            name=parameter,
        )

    return sampled_params


def clean_infection_data_measure(infection_data: pd.DataFrame, measure: str) -> pd.Series:
    data = infection_data[measure].dropna()
    min_date = data.reset_index().groupby('location_id').date.min()
    prepend_date = min_date - pd.Timedelta(days=1)
    prepend_idx = prepend_date.reset_index().set_index(['location_id', 'date']).index
    prepend = pd.Series(0., index=prepend_idx, name=measure)
    data = data.append(prepend).sort_index().reset_index()

    all_locs = data.location_id.unique().tolist()
    global_date_range = pd.date_range(data.date.min(), data.date.max())
    square_idx = pd.MultiIndex.from_product((all_locs, global_date_range), names=['location_id', 'date']).sort_values()
    data = data.set_index(['location_id', 'date']).reindex(square_idx).groupby('location_id').bfill()
    return data[measure]


def make_initial_condition(parameters: Parameters, population: pd.DataFrame):
    # Alpha is time-invariant
    alpha = parameters.alpha_all.groupby('location_id').first()

    group_pop = get_risk_group_pop(population)
    # Filter out early dates with few infections
    # to reduce noise in the past fit from leaking into the beta regression.
    infections = parameters.new_e_all.groupby('location_id').apply(filter_to_epi_threshold)
    infections_by_group = group_pop.div(group_pop.sum(axis=1), axis=0).mul(infections, axis=0)
    new_e_start = infections_by_group.reset_index(level='date').groupby('location_id').first()
    start_date, new_e_start = new_e_start['date'], new_e_start[list(RISK_GROUP_NAMES)]

    compartments = [f'{compartment}_{risk_group}'
                    for risk_group, compartment
                    in itertools.product(RISK_GROUP_NAMES, COMPARTMENTS_NAMES + TRACKING_COMPARTMENTS_NAMES)]
    initial_condition = pd.DataFrame(0., columns=compartments, index=parameters.new_e_all.index)
    for location_id, loc_start_date in start_date.iteritems():
        for risk_group in RISK_GROUP_NAMES:
            pop = group_pop.loc[location_id, risk_group]
            new_e = new_e_start.loc[location_id, risk_group]
            suffix = f'_unvaccinated_{risk_group}'
            # Backfill everyone susceptible
            initial_condition.loc[pd.IndexSlice[location_id, :loc_start_date], f'S_none{suffix}'] = pop
            # Set initial condition on start date
            infectious = (new_e / 5) ** (1 / alpha.loc[location_id])
            initial_condition.loc[(location_id, loc_start_date), f'S_none{suffix}'] = pop - new_e - infectious
            initial_condition.loc[(location_id, loc_start_date), f'E_ancestral{suffix}'] = new_e
            initial_condition.loc[(location_id, loc_start_date), f'NewE_ancestral{suffix}'] = new_e
            initial_condition.loc[(location_id, loc_start_date), f'I_ancestral{suffix}'] = infectious
            for variant in VARIANT_NAMES:
                initial_condition.loc[pd.IndexSlice[location_id, :loc_start_date], f'EffectiveSusceptible_{variant}{suffix}'] = pop
            
    return initial_condition


def get_risk_group_pop(population: pd.DataFrame):
    population_low_risk = (population[population['age_group_years_start'] < 65]
                           .groupby('location_id')['population']
                           .sum()
                           .rename('lr'))
    population_high_risk = (population[population['age_group_years_start'] >= 65]
                            .groupby('location_id')['population']
                            .sum()
                            .rename('hr'))
    pop = pd.concat([population_low_risk, population_high_risk], axis=1)
    return pop


def filter_to_epi_threshold(infections: pd.Series,
                            threshold: float = 50.) -> pd.Series:
    infections = infections.reset_index(level='location_id', drop=True)
    # noinspection PyTypeChecker
    start_date = infections.loc[threshold <= infections].index.min()
    while infections.loc[start_date:].dropna().count() <= 2:
        threshold *= 0.5
        # noinspection PyTypeChecker
        start_date = infections.loc[threshold <= infections].index.min()
        if threshold < 1e-6:
            start_date = infections.index.min()
            break
    return infections.loc[start_date:]


def run_ode_fit(initial_condition: pd.DataFrame, ode_parameters: Parameters):
    full_compartments = solver.run_ode_model(
        initial_condition,
        *ode_parameters.to_dfs(),
        forecast=False,
        num_cores=5,
    )
    import pdb; pdb.set_trace()
    betas = []
    all_compartments = [f"{c}_{rg}" for c, rg in itertools.product(COMPARTMENTS_NAMES, RISK_GROUP_NAMES)]
    population = full_compartments.loc[:, all_compartments].sum(axis=1)

    total_infectious = 0.
    beta_fit = 0.

    for risk_group in RISK_GROUP_NAMES:
        for variant_name, variant_index in VARIANT._asdict().items():
            new_e = (full_compartments
                     .filter(like=f'NewE_{variant_name}')
                     .sum(axis=1)
                     .groupby('location_id')
                     .diff()
                     .fillna(0))

    #     variant_indices = CG_SUSCEPTIBLE(variant_index)
    #     variant_indices = np.hstack([variant_indices, variant_indices + system_size]).tolist()
    #     susceptible_variant = full_compartments.iloc[:, variant_indices].sum(axis=1)
    #
    #     variant_indices = CG_INFECTIOUS(variant_index)
    #     variant_indices = np.hstack([variant_indices, variant_indices + system_size]).tolist()
    #     infectious_variant = full_compartments.iloc[:, variant_indices].sum(axis=1)
    #     total_infectious += infectious_variant
    #
    #     disease_density = susceptible_variant * infectious_variant**ode_parameters.alpha / population
    #     beta_variant = (new_e_variant / disease_density).rename(f'beta_{variant_name}')
    #     beta_variant.loc[~disease_density.isnull()] = beta_variant.loc[~disease_density.isnull()].fillna(0.)
    #     betas.append(beta_variant)
    #
    #     beta_fit += beta_variant / parameter_df[f'kappa_{variant_name}'] * infectious_variant
    # beta_fit /= total_infectious
    # betas = pd.concat([beta_fit.rename('beta')] + betas, axis=1)
    # betas.loc[betas.beta.isnull(), :] = np.nan
    pass
#    return betas, full_compartments

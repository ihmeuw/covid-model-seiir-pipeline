import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT,
    RISK_GROUP,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)


def prepare_ode_fit_parameters(past_infections: pd.Series,
                               rhos: pd.DataFrame,
                               vaccinations: pd.DataFrame,
                               sampled_params: Dict[str, pd.Series]) -> Parameters:
    past_index = past_infections.index
    rhos = rhos.reindex(past_index, fill_value=0.).to_dict('series')
    betas = {f'beta_{variant}': pd.Series(np.nan, index=past_index, name=f'beta_{variant}') for variant in VARIANT}
    vaccinations = vaccinations.reindex(past_index, fill_value=0.).to_dict('series')

    return Parameters(
        **sampled_params,
        new_e=past_infections,
        **betas,
        **rhos,
        **vaccinations,
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
    return data.infections


def make_initial_condition(parameters: Parameters, population: pd.DataFrame):
    # Alpha is time-invariant
    alpha = parameters.alpha.groupby('location_id').first()

    group_pop = get_risk_group_pop(population)
    # Filter out early dates with few infections
    # to reduce noise in the past fit from leaking into the beta regression.
    infections = parameters.new_e.groupby('location_id').apply(filter_to_epi_threshold)
    infections_by_group = group_pop.div(group_pop.sum(axis=1), axis=0).mul(infections, axis=0)
    new_e_start = infections_by_group.reset_index(level='date').groupby('location_id').first()
    start_date, new_e_start = new_e_start['date'], new_e_start[list(RISK_GROUP)]

    compartments = [f'{compartment}_{risk_group}'
                    for risk_group, compartment
                    in itertools.product(RISK_GROUP, COMPARTMENTS_NAMES + TRACKING_COMPARTMENTS_NAMES)]
    initial_condition = pd.DataFrame(0., columns=compartments, index=parameters.new_e.index)
    for location_id, loc_start_date in start_date.iteritems():
        for risk_group in RISK_GROUP:
            pop = group_pop.loc[location_id, risk_group]
            new_e = new_e_start.loc[location_id, risk_group]
            suffix = f'_unvaccinated_{risk_group}'
            # Backfill everyone susceptible
            initial_condition.loc[pd.IndexSlice[location_id, :loc_start_date], f'S_unprotected{suffix}'] = pop
            # Set initial condition on start date
            infectious = (new_e / 5) ** (1 / alpha.loc[location_id])
            initial_condition.loc[(location_id, loc_start_date), f'S_unprotected{suffix}'] = pop - new_e - infectious
            initial_condition.loc[(location_id, loc_start_date), f'E_ancestral{suffix}'] = new_e
            initial_condition.loc[(location_id, loc_start_date), f'NewE_unvaccinated_{risk_group}'] = new_e
            initial_condition.loc[(location_id, loc_start_date), f'NewE_ancestral_{risk_group}'] = new_e
            initial_condition.loc[(location_id, loc_start_date), f'I_ancestral{suffix}'] = infectious
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


def run_ode_fit(start_dates: pd.Series,
                initial_condition: pd.DataFrame,
                ode_parameters: Parameters,):
    past_index = ode_parameters.new_e.index



def run_ode_fit(ode_parameters: Parameters,
                progress_bar: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fit_results = []
    ode_parameter_list = tqdm.tqdm(list(ode_parameters), disable=not progress_bar)
    for location_id, location_params in ode_parameter_list:
        loc_fit_results = run_loc_ode_fit(
            location_params
        )
        loc_fit_results['location_id'] = location_id
        loc_fit_results = loc_fit_results.set_index(['location_id', 'date'])

        fit_results.append(loc_fit_results)
    fit_results = pd.concat(fit_results).sort_index()
    return fit_results[['beta', 'beta_wild', 'beta_variant']], fit_results[[c for c in fit_results if 'beta' not in c]]


def run_loc_ode_fit(ode_parameters: Parameters) -> pd.DataFrame:
    date = pd.Series(infections.index.values)
    t = (date - date.min()).dt.days.values
    obs = infections.values






    params = np.hstack(params).T

    result = math.solve_ode(
        system=ode.fit_system,
        t=t,
        init_cond=initial_condition,
        params=params,
    )
    components = pd.DataFrame(
        data=result.T,
        columns=[f'{compartment}_{risk_group}'
                 for risk_group, compartment in itertools.product(pop_groups, all_compartments)],
    )
    components['date'] = date
    components = components.set_index('date')

    new_e_wild = components.filter(like='NewE_wild').sum(axis=1)
    new_e_variant = components.filter(like='NewE_variant').sum(axis=1)

    s_wild_compartments = [f'{c}_{r}' for c, r in itertools.product(['S', 'S_u', 'S_p', 'S_pa'], pop_groups)]
    s_wild = components.loc[:, s_wild_compartments].sum(axis=1)
    s_variant_compartments = [f'{c}_{r}' for c, r
                              in itertools.product(['S_variant', 'S_variant_u', 'S_variant_pa', 'S_m'], pop_groups)]
    s_variant_only = components.loc[:, s_variant_compartments].sum(axis=1)
    s_variant = s_wild + s_variant_only
    i_wild = components.loc[:, [c for c in components if c[0] == 'I' and 'variant' not in c]].sum(axis=1)
    i_variant = components.loc[:, [c for c in components if c[0] == 'I' and 'variant' in c]].sum(axis=1)

    disease_density_wild = s_wild * i_wild**ode_parameters.alpha.values / pop
    beta_wild = (new_e_wild.diff() / disease_density_wild).reindex(full_index)
    disease_density_variant = s_variant * i_variant**ode_parameters.alpha.values / pop
    beta_variant = (new_e_variant.diff() / disease_density_variant).reindex(full_index)

    components = components.reindex(full_index, fill_value=0.)
    for risk_group, group_pop in pop_groups.items():
        components.loc[components[f'S_{risk_group}'] == 0, f'S_{risk_group}'] = group_pop

    components['beta_wild'] = beta_wild
    components['beta_variant'] = beta_variant

    rho = ode_parameters.rho
    rho_b1617 = ode_parameters.rho_b1617
    kappa = ode_parameters.kappa
    phi = ode_parameters.phi
    psi = ode_parameters.psi
    beta1 = beta_wild / (1 + kappa * rho)
    beta2 = beta_variant / (1 + kappa * (phi * (1 - rho_b1617) + rho_b1617 * psi))
    components['beta'] = (i_wild * beta1 + (i_variant * beta2).fillna(0)) / (i_wild + i_variant)
    if np.any(components['beta'] < 0):
        raise ValueError('Negative betas found in the fit results. Check that cumulative '
                         'infected does not exceed 100% of the population.')
    return components.reset_index()







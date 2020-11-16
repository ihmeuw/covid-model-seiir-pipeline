from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.model.ode_forecast import CompartmentInfo




def compute_output_metrics(infection_data: pd.DataFrame,
                           ifr: pd.DataFrame,
                           components_past: pd.DataFrame,
                           components_forecast: pd.DataFrame,
                           seir_params: Dict[str, float],
                           compartment_info: CompartmentInfo) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = splice_components(components_past, components_forecast)

    observed_infections, observed_deaths = get_observed_infecs_and_deaths(infection_data)
    infection_death_lag = infection_data['i_d_lag'].max()
    if compartment_info.group_suffixes:
        modeled_infections, modeled_deaths = 0, 0
        for group in compartment_info.group_suffixes:
            group_compartments = [c for c in compartment_info.compartments if group in c]
            group_infections = compute_infections(components[['date'] + group_compartments])
            group_ifr = ifr[f'ifr_{group}'].rename('ifr')
            group_deaths = compute_deaths(group_infections, infection_death_lag, group_ifr)
            modeled_infections += group_infections
            modeled_deaths += group_deaths
    else:
        modeled_infections = compute_infections(components[['date'] + compartment_info.compartments])
        modeled_deaths = compute_deaths(modeled_infections, infection_death_lag, ifr['ifr'])
    modeled_infections = modeled_infections.to_frame()
    modeled_deaths = modeled_deaths.reset_index(level='observed')
    import pdb; pdb.set_trace()
    infections = observed_infections.combine_first(modeled_infections)
    deaths = observed_deaths.combine_first(modeled_deaths)
    r_effective = compute_effective_r(infection_data, components, seir_params, seiir_compartments)

    return components, infections, deaths, r_effective


def splice_components(components_past: pd.DataFrame, components_forecast: pd.DataFrame):
    components_past = components_past.reindex(components_forecast.columns, axis='columns').reset_index()
    components_forecast = components_forecast.reset_index()
    components = (pd.concat([components_past, components_forecast])
                  .sort_values(['location_id', 'date'])
                  .set_index(['location_id']))
    return components


def get_observed_infecs_and_deaths(infection_data: pd.DataFrame):
    observed = infection_data['obs_infecs'] == 1
    observed_infections = (infection_data
                           .loc[observed, ['location_id', 'date', 'cases_draw']]
                           .set_index(['location_id', 'date'])
                           .sort_index()
                           .rename(columns={'cases_draw': 'infections'}))
    observed = infection_data['obs_deaths'] == 1
    observed_deaths = (infection_data
                       .loc[observed, ['location_id', 'date', 'deaths_mean']]
                       .rename(columns={'deaths_mean': 'deaths'})
                       .set_index(['location_id', 'date'])
                       .sort_index())
    observed_deaths['observed'] = 1
    return observed_infections, observed_deaths


def compute_infections(components: pd.DataFrame) -> pd.Series:
    delta_susceptible = (components[[c for c in components.columns if 'S' in c]]
                         .sum(axis=1)
                         .groupby('location_id')
                         .apply(lambda x: x.shift(1) - x)
                         .fillna(0)
                         .rename('infections'))
    immune_cols = [c for c in components.columns if 'M' in c]
    if immune_cols:
        immune = (components[immune_cols]
                  .sum(axis=1)
                  .groupby('location_id')
                  .apply(lambda x: x.shift(1) - x)
                  .fillna(0)
                  .rename('infections'))
    else:
        immune = 0

    modeled_infections = delta_susceptible - immune
    modeled_infections = (pd.concat([components['date'], modeled_infections], axis=1)
                          .reset_index()
                          .set_index(['location_id', 'date'])
                          .sort_index()['infections'])

    return modeled_infections


def compute_deaths(modeled_infections: pd.Series, infection_death_lag: int, ifr: pd.Series) -> pd.Series:
    modeled_deaths = (modeled_infections.shift(infection_death_lag).dropna() * ifr).rename('deaths').reset_index()
    modeled_deaths['observed'] = 0
    modeled_deaths = modeled_deaths.set_index(['location_id', 'date', 'observed'])['deaths']
    return modeled_deaths


def compute_effective_r(infection_data: pd.DataFrame, components: pd.DataFrame,
                        beta_params: Dict[str, float],
                        seiir_compartments: List[str]) -> pd.DataFrame:
    alpha, sigma = beta_params['alpha'], beta_params['sigma']
    gamma1, gamma2 = beta_params['gamma1'], beta_params['gamma2']

    components = components.reset_index().set_index(['location_id', 'date'])

    beta, theta = components['beta'], components['theta']
    s = components[[c for c in seiir_compartments if 'S' in c]].sum(axis=1)
    i1 = components[[c for c in seiir_compartments if 'I1' in c]].sum(axis=1)
    i2 = components[[c for c in seiir_compartments if 'I2' in c]].sum(axis=1)
    n = infection_data.groupby('location_id')['pop'].max()
    avg_gamma = 1 / (1 / (gamma1*(sigma - theta)) + 1 / (gamma2*(sigma - theta)))

    r_controlled = beta * alpha * sigma / avg_gamma * (i1 + i2) ** (alpha - 1)
    r_effective = (r_controlled * s / n).rename('r_effective')

    return r_effective

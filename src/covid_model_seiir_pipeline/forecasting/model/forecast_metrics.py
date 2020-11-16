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
    import pdb; pdb.set_trace()
    
    observed = infection_data['obs_infecs'] == 1
    observed_infections = (infection_data
                           .loc[observed, ['location_id', 'date', 'cases_draw']]
                           .set_index(['location_id', 'date'])
                           .sort_index())
    


    if compartment_info.group_suffixes:
        for group in compartment_info.group_suffixes:
            group_compartments = [c for c in compartment_info.compartments if group in c]
            observed_infections, modeled_infections = compute_infections(infection_data, components)
    observed_deaths, modeled_deaths = compute_deaths(infection_data, modeled_infections)

    infections = observed_infections.combine_first(modeled_infections)['cases_draw'].rename('infections')
    deaths = observed_deaths.combine_first(modeled_deaths).rename(columns={'deaths_draw': 'deaths'})
    r_effective = compute_effective_r(infection_data, components, seir_params, seiir_compartments)

    return components, infections, deaths, r_effective


def splice_components(components_past: pd.DataFrame, components_forecast: pd.DataFrame):
    components_past = components_past.reindex(components_forecast.columns, axis='columns').reset_index()
    components_forecast = components_forecast.reset_index()
    components = (pd.concat([components_past, components_forecast])
                  .sort_values(['location_id', 'date'])
                  .set_index(['location_id']))
    return components


def compute_infections(infection_data: pd.DataFrame,
                       components: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    delta_susceptible = (components[[c for c in components.columns if 'S' in c]]
                         .sum(axis=1)
                         .groupby('location_id')
                         .apply(lambda x: x.shift(1) - x)
                         .fillna(0)
                         .rename('cases_draw'))
    immune_cols = [c for c in components.columns if 'M' in c]
    if immune_cols:
        immune = (components[immune_cols]
                  .sum(axis=1)
                  .groupby('location_id')
                  .apply(lambda x: x.shift(1) - x)
                  .fillna(0)
                  .rename('cases_draw'))
    else:
        immune = 0

    modeled_infections = delta_susceptible - immune
    modeled_infections = (pd.concat([components['date'], modeled_infections], axis=1)
                          .reset_index()
                          .set_index(['location_id', 'date'])
                          .sort_index())

    return observed_infections, modeled_infections


def compute_deaths(infection_data: pd.DataFrame,
                   modeled_infections: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = infection_data['obs_deaths'] == 1
    observed_deaths = (infection_data
                       .loc[observed, ['location_id', 'date', 'deaths_mean']]
                       .rename(columns={'deaths_mean': 'deaths_draw'})
                       .set_index(['location_id', 'date'])
                       .sort_index())
    observed_deaths['observed'] = 1

    infection_death_lag = infection_data['i_d_lag'].max()

    def _compute_ifr(data):
        data = data.set_index('date')
        has_deaths = data['obs_deaths'] == 1
        deaths = data['deaths_draw']
        infecs = data['cases_draw']
        return ((deaths / infecs.shift(infection_death_lag))
                .loc[has_deaths]
                .dropna()
                .rename('ifr'))

    ifr = (infection_data
           .groupby('location_id')
           .apply(_compute_ifr))

    modeled_deaths = modeled_infections['cases_draw'].shift(infection_death_lag).dropna()
    modeled_deaths = pd.concat([modeled_deaths, ifr], axis=1).reset_index()
    modeled_deaths['ifr'] = (modeled_deaths
                             .groupby('location_id')['ifr']
                             .apply(lambda x: x.fillna(method='pad')))
    modeled_deaths['deaths_draw'] = modeled_deaths['cases_draw'] * modeled_deaths['ifr']
    modeled_deaths = (modeled_deaths
                      .loc[:, ['location_id', 'date', 'deaths_draw']]
                      .set_index(['location_id', 'date'])
                      .fillna(0))

    modeled_deaths['observed'] = 0
    return observed_deaths, modeled_deaths


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

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_output_metrics(infection_data: pd.DataFrame,
                           components_past: pd.DataFrame,
                           components_forecast: pd.DataFrame,
                           thetas: pd.Series,
                           seir_params: Dict[str, float],
                           ode_system: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = splice_components(components_past, components_forecast)
    components['theta'] = thetas.reindex(components.index).fillna(0)

    observed_infections, modeled_infections = compute_infections(infection_data, components)
    observed_deaths, modeled_deaths = compute_deaths(infection_data, modeled_infections)

    infections = observed_infections.combine_first(modeled_infections)['cases_draw'].rename('infections')
    deaths = observed_deaths.combine_first(modeled_deaths).rename(columns={'deaths_draw': 'deaths'})
    r_effective = compute_effective_r(infection_data, components, seir_params, ode_system)

    return components, infections, deaths, r_effective


def splice_components(components_past: pd.DataFrame, components_forecast: pd.DataFrame):
    shared_columns = ['date', 'S', 'E', 'I1', 'I2', 'R', 'beta', 'theta']
    components_past['theta'] = np.nan
    components_past = components_past[shared_columns].reset_index()
    components_forecast = components_forecast[['location_id'] + shared_columns]
    components = (pd.concat([components_past, components_forecast])
                  .sort_values(['location_id', 'date'])
                  .set_index(['location_id']))
    return components


def compute_infections(infection_data: pd.DataFrame,
                       components: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = infection_data['obs_infecs'] == 1
    observed_infections = (infection_data
                           .loc[observed, ['location_id', 'date', 'cases_draw']]
                           .set_index(['location_id', 'date'])
                           .sort_index())

    modeled_infections = (components
                          .groupby('location_id')['S']
                          .apply(lambda x: x.shift(1) - x)
                          .fillna(0)
                          .rename('cases_draw'))
    modeled_infections = pd.concat([components['date'], modeled_infections], axis=1).reset_index()
    modeled_infections = modeled_infections.set_index(['location_id', 'date']).sort_index()
    return observed_infections, modeled_infections


def compute_deaths(infection_data: pd.DataFrame,
                   modeled_infections: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = infection_data['obs_deaths'] == 1
    observed_deaths = (infection_data
                       .loc[observed, ['location_id', 'date', 'deaths_draw']]
                       .set_index(['location_id', 'date'])
                       .sort_index())
    observed_deaths['observed'] = 1

    infection_death_lag = infection_data['i_d_lag'].max()

    def _compute_ifr(data):
        deaths = data.set_index('date')['deaths_draw']
        infecs = data.set_index('date')['cases_draw']
        return (deaths / infecs.shift(infection_death_lag)).dropna().rename('ifr')

    ifr = (infection_data
           .loc[observed]
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
                        beta_params: Dict[str, float], ode_system: str) -> pd.DataFrame:
    alpha, sigma = beta_params['alpha'], beta_params['sigma']
    gamma1, gamma2 = beta_params['gamma1'], beta_params['gamma2']

    components = components.reset_index().set_index(['location_id', 'date'])

    beta, theta = components['beta'], components['theta']
    s, i1, i2 = components['S'], components['I1'], components['I2']
    n = infection_data.groupby('location_id')['pop'].max()

    if ode_system == 'old_theta':
        avg_gamma = 1 / (1 / gamma1 + 1 / gamma2)
        r_controlled = beta * alpha / avg_gamma * (i1 + i2) ** (alpha - 1)
    elif ode_system == 'new_theta':
        avg_gamma = 1 / (1 / (gamma1*(sigma - theta)) + 1 / (gamma2*(sigma - theta)))
        r_controlled = beta * alpha * sigma / avg_gamma * (i1 + i2) ** (alpha - 1)
    else:
        raise NotImplementedError(f'Unknown ode system type {ode_system}.')

    r_effective = (r_controlled * s / n).rename('r_effective')
    return r_effective

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
    shared_columns = ['date',
                      'S', 'E', 'I1', 'I2', 'R',
                      'S_v', 'E_v', 'I1_v', 'I2_v', 'R_v', 'R_sv',
                      'beta', 'theta']
    shared_columns = [i for i in shared_columns if i in components_past.columns and i in components_forecast.columns]
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

    if 'S_v' in components.columns:
        components = components.copy()
        components['S_'] = components[['S', 'S_v']].sum(axis=1)
        s = (components
               .groupby('location_id')['S_']
               .apply(lambda x: x.shift(1) - x)
               .fillna(0)
               .rename('cases_draw'))
        r_sv = (components
             .groupby('location_id')['R_sv']
             .apply(lambda x: x.shift(1) - x)
             .fillna(0)
             .rename('cases_draw'))
        modeled_infections = s + r_sv
    else:
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
                       .loc[observed, ['location_id', 'date', 'deaths_mean']]
                       .rename(columns={'deaths_mean': 'deaths_draw'})
                       .set_index(['location_id', 'date'])
                       .sort_index())
    observed_deaths['observed'] = 1

    infection_death_lag = infection_data['i_d_lag'].max()

    def _compute_ifr(data):
        deaths = data['deaths_draw']
        infecs = data['cases_draw']
        return (deaths / infecs.shift(infection_death_lag)).dropna().mean()

    ifr = (infection_data
           .groupby('location_id')
           .apply(_compute_ifr))
    modeled_deaths = ((modeled_infections['cases_draw'] * ifr)
                      .shift(infection_death_lag)
                      .rename('deaths_draw')
                      .to_frame())
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
    
    if ode_system == 'vaccine':
        s_v, i1_v, i2_v = components['S_v'], components['I1_v'], components['I2_v']
        s += s_v
        i1 += i1_v
        i2 += i2_v

    if ode_system == 'old_theta':
        avg_gamma = 1 / (1 / gamma1 + 1 / gamma2)
        r_controlled = beta * alpha / avg_gamma * (i1 + i2) ** (alpha - 1)
    elif ode_system in ['new_theta', 'vaccine']:
        avg_gamma = 1 / (1 / (gamma1*(sigma - theta)) + 1 / (gamma2*(sigma - theta)))
        r_controlled = beta * alpha * sigma / avg_gamma * (i1 + i2) ** (alpha - 1)
    else:
        raise NotImplementedError(f'Unknown ode system type {ode_system}.')

    r_effective = (r_controlled * s / n).rename('r_effective')
    return r_effective

from typing import Dict, Tuple, List

from covid_model_seiir_pipeline import static_vars

import numpy as np
import pandas as pd


def compute_output_metrics(infection_data: pd.DataFrame,
                           components_past: pd.DataFrame,
                           components_forecast: pd.DataFrame,
                           thetas: pd.Series,
                           vaccinations: pd.DataFrame,
                           seir_params: Dict[str, float],
                           ode_system: str,
                           seiir_compartments: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = splice_components(components_past, components_forecast, seiir_compartments)
    components['theta'] = thetas.reindex(components.index).fillna(0)
    if vaccinations is not None:
        components = components.reset_index().merge(vaccinations.reset_index(), how='left').set_index('location_id')

    observed_infections, modeled_infections = compute_infections(infection_data, components, seiir_compartments)
    observed_deaths, modeled_deaths = compute_deaths(infection_data, modeled_infections)

    infections = observed_infections.combine_first(modeled_infections)['cases_draw'].rename('infections')
    deaths = observed_deaths.combine_first(modeled_deaths).rename(columns={'deaths_draw': 'deaths'})
    r_effective = compute_effective_r(infection_data, components, seir_params, ode_system, seiir_compartments)

    return components, infections, deaths, r_effective


def splice_components(components_past: pd.DataFrame, components_forecast: pd.DataFrame, seiir_compartments: List[str]):
    shared_columns = ['date', 'beta', 'theta'] + seiir_compartments
    shared_columns = [i for i in shared_columns if i in components_past.columns and i in components_forecast.columns]
    components_past['theta'] = np.nan
    components_past = components_past[shared_columns].reset_index()
    components_forecast = components_forecast[['location_id'] + shared_columns]
    components = (pd.concat([components_past, components_forecast])
                  .sort_values(['location_id', 'date'])
                  .set_index(['location_id']))
    return components


def compute_infections(infection_data: pd.DataFrame,
                       components: pd.DataFrame,
                       seiir_compartments: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = infection_data['obs_infecs'] == 1
    observed_infections = (infection_data
                           .loc[observed, ['location_id', 'date', 'cases_draw']]
                           .set_index(['location_id', 'date'])
                           .sort_index())

    if set(seiir_compartments) == set(static_vars.VACCINE_SEIIR_COMPARTMENTS):
        modeled_infections = (components
                              .groupby('location_id')['S']
                              .apply(lambda x: x.shift(1) - x)
                              .fillna(0)
                              .rename('cases_draw'))
    else:
        components = components.copy()
        components['S_'] = components[[c for c in seiir_compartments if 'S' in c]].sum(axis=1)
        components['R_sv_'] = components[[c for c in seiir_compartments if 'R_sv' in c]].sum(axis=1)
        s = (components
               .groupby('location_id')['S_']
               .apply(lambda x: x.shift(1) - x)
               .fillna(0)
               .rename('cases_draw'))
        r_sv = (components
             .groupby('location_id')['R_sv_']
             .apply(lambda x: x.shift(1) - x)
             .fillna(0)
             .rename('cases_draw'))
        modeled_infections = s + r_sv
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
                        beta_params: Dict[str, float], ode_system: str,
                        seiir_compartments: List[str]) -> pd.DataFrame:
    alpha, sigma = beta_params['alpha'], beta_params['sigma']
    gamma1, gamma2 = beta_params['gamma1'], beta_params['gamma2']

    components = components.reset_index().set_index(['location_id', 'date'])

    beta, theta = components['beta'], components['theta']
    s = components[[c for c in seiir_compartments if 'S' in c]].sum(axis=1)
    i1 = components[[c for c in seiir_compartments if 'I1' in c]].sum(axis=1)
    i2 = components[[c for c in seiir_compartments if 'I2' in c]].sum(axis=1)
    n = infection_data.groupby('location_id')['pop'].max()

    if ode_system in ['normal', 'vaccine']:
        avg_gamma = 1 / (1 / (gamma1*(sigma - theta)) + 1 / (gamma2*(sigma - theta)))
        r_controlled = beta * alpha * sigma / avg_gamma * (i1 + i2) ** (alpha - 1)
    else:
        raise NotImplementedError(f'Unknown ode system type {ode_system}.')

    r_effective = (r_controlled * s / n).rename('r_effective')
    return r_effective

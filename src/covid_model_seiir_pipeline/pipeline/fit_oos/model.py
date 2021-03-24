from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.pipeline.fit_oos.containers import (
    ODEParameters,
)
from covid_model_seiir_pipeline.pipeline.fit_oos.ode_system import (
    compartments,
    parameters,
    vaccine_types,
    single_force_system,
    ramp_force_system,
)


def prepare_ode_fit_parameters(past_infections: pd.Series,
                               population: pd.Series,
                               vaccinations: pd.DataFrame,
                               variant_prevalence: pd.DataFrame,
                               fit_parameters: Dict,
                               draw_id: int) -> ODEParameters:
    past_index = past_infections.index
    population = population.reindex(past_index, level='location_id')

    np.random.seed(draw_id)
    sampled_params = {}
    for parameter in ['alpha', 'sigma', 'gamma1', 'gamma2']:
        sampled_params[parameter] = pd.Series(
            np.random.uniform(*fit_parameters[parameter]),
            index=past_index,
            name=parameter,
        )
    for parameter in ['kappa', 'phi', 'pi', 'epsilon', 'a', 'b', 'p_cross_immune']:
        sampled_params[parameter] = pd.Series(
            fit_parameters[parameter],
            index=past_index,
            name=parameter,
        )

    vaccinations = math.adjust_vaccinations(vaccinations)
    vaccinations = pd.concat([v.rename(k) for k, v in vaccinations.items()], axis=1)
    vaccinations = vaccinations.reindex(past_index, fill_value=0.)

    return ODEParameters(
        system=fit_parameters['system'],
        population=population,
        new_e=past_infections,
        **sampled_params,
        rho=variant_prevalence['rho'].reindex(past_index, fill_value=0),
        rho_variant=variant_prevalence['rho_variant'].reindex(past_index, fill_value=0),
        vaccines_unprotected=vaccinations.filter(like='unprotected').sum(axis=1),
        vaccines_protected_wild_type=vaccinations.filter(like='protected_wild').sum(axis=1),
        vaccines_protected_all_types=vaccinations.filter(like='protected_all').sum(axis=1),
        vaccines_immune_wild_type=vaccinations.filter(like='immune_wild').sum(axis=1),
        vaccines_immune_all_types=vaccinations.filter(like='immune_all').sum(axis=1),
    )


def clean_infection_data_measure(infection_data: pd.DataFrame, measure: str) -> pd.Series:
    """Extracts measure, drops nulls, adds a leading zero.

    Infections and deaths have a non-overlapping past index due to the way
    the infections ES is built. This function, pulls out a measure, drops
    nulls from the non-overlaping region, and then pads the front of the
    series with a 0 so that the resulting series has the property:

        s == s.groupby('location_id').cumsum().groupby('location_id').diff().fillna(0)

    which is to say we can preserve the counts under conversions between daily
    and cumulative space.

    """
    data = infection_data[measure].dropna()
    min_date = data.reset_index().groupby('location_id').date.min()
    prepend_date = min_date - pd.Timedelta(days=1)
    prepend_idx = prepend_date.reset_index().set_index(['location_id', 'date']).index
    prepend = pd.Series(0., index=prepend_idx, name=measure)
    return data.append(prepend).sort_index()


def run_ode_fit(ode_parameters: ODEParameters, progress_bar: bool) -> Tuple[pd.Series, pd.DataFrame]:
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


def run_loc_ode_fit(ode_parameters: ODEParameters) -> pd.DataFrame:
    # Filter out early dates with few infections
    # to reduce noise in the past fit from leaking into the beta regression.
    infections = ode_parameters.new_e
    full_index = infections.index
    infections = filter_to_epi_threshold(infections)
    ode_parameters = ode_parameters.reindex(infections.index)

    date = pd.Series(infections.index.values)
    t = (date - date.min()).dt.days.values
    obs = infections.values
    total_population = ode_parameters.population.iloc[0]

    initial_condition = np.zeros(len(compartments))
    initial_condition[compartments.S] = total_population - obs[0] - (obs[0] / 5) ** (1.0 / ode_parameters.alpha[0])
    initial_condition[compartments.E] = obs[0]
    initial_condition[compartments.I1] = (obs[0] / 5) ** (1.0 / ode_parameters.alpha[0])

    params = np.hstack([
        ode_parameters.to_df().loc[:, list(parameters._fields)].values.T,
        ode_parameters.get_vaccinations(vaccine_types._fields).values.T,
    ])

    system = {
        'single': single_force_system,
        'ramp': ramp_force_system,
    }[ode_parameters.system]

    result = math.solve_ode(
        system=system,
        t=t,
        init_cond=initial_condition,
        params=params,
    )
    components = pd.DataFrame(
        data=result.T,
        columns=compartments._fields,
    )
    components['date'] = date
    components = components.set_index('date')

    assert (components['S'] >= 0.0).all()

    new_e_wild = components['NewE_wild']
    new_e_variant = components['NewE_variant']

    assert np.allclose(obs, new_e_wild + new_e_variant)

    s_wild = components.loc[:, [c for c in components if c in ['S', 'S_u', 'S_p', 'S_pa']]].sum(axis=1)
    s_variant_only = (components
                      .loc[:, [c for c in components if c in ['S_variant', 'S_variant_u', 'S_varaint_pa', 'S_m']]]
                      .sum(axis=1))
    s_variant = s_wild + s_variant_only
    i_wild = components.loc[:, [c for c in components if c[0] == 'I' and 'variant' not in c]].sum(axis=1)
    i_variant = components.loc[:, [c for c in components if c[0] == 'I' and 'variant' in c]].sum(axis=1)

    disease_density_wild = s_wild * i_wild**ode_parameters.alpha.values / total_population
    beta_wild = (new_e_wild / disease_density_wild).reindex(full_index)
    disease_density_variant = s_variant * i_variant**ode_parameters.alpha.values / total_population
    beta_variant = (new_e_variant / disease_density_variant).reindex(full_index)
    disease_density_all = (s_wild + s_variant) * (i_wild + i_variant)**ode_parameters.alpha.values / total_population
    beta = (obs / disease_density_all).reindex(full_index)

    components = components.reindex(full_index, fill_value=0.)
    components.loc[components['S'] == 0, 'S'] = total_population
    components['beta'] = beta
    components['beta_wild'] = beta_wild
    components['beta_variant'] = beta_variant

    return components.reset_index()


def filter_to_epi_threshold(infections: pd.Series,
                            threshold: float = 50.) -> pd.Series:
    # noinspection PyTypeChecker
    start_date = infections.loc[threshold <= infections].index.min()
    while infections.loc[start_date:].count() <= 2:
        threshold *= 0.5
        # noinspection PyTypeChecker
        start_date = infections.loc[threshold <= infections].index.min()
        if threshold < 1e-6:
            start_date = infections.index.min()
            break
    return infections.loc[start_date:]

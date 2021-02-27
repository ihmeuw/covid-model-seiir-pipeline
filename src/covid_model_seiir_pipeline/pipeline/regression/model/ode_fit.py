from typing import Dict

import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.pipeline.regression.model.containers import (
    ODEParameters,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    past_system,
)


def prepare_ode_fit_parameters(past_index: pd.Index,
                               population: pd.Series,
                               vaccinations: pd.DataFrame,
                               regression_parameters: Dict,
                               draw_id: int) -> ODEParameters:
    population = population.reindex(past_index, level='location_id')
    np.random.seed(draw_id)
    sampled_params = {}
    for parameter in ['alpha', 'sigma', 'gamma1', 'gamma2']:
        sampled_params[parameter] = pd.Series(
            np.random.uniform(*regression_parameters[parameter]),
            index=past_index,
            name=parameter,
        )

    # TODO: test out vaccine system.
    ready_to_switch = False
    if not ready_to_switch:
        vaccines_immune = pd.Series(0., index=past_index, name='vaccines_immune')
        vaccines_other = pd.Series(0., index=past_index, name='vaccines_other')
    else:
        vaccinations = vaccinations.reindex(past_index, fill_value=0)
        vaccines_all = vaccinations.sum(axis=1).loc[past_index]
        vaccines_immune = vaccinations[[c for c in vaccinations
                                        if 'effective' in c and 'protected' not in c]].sum(axis=1)
        vaccines_other = vaccines_all - vaccines_immune

    return ODEParameters(
        population=population,
        **sampled_params,
        vaccines_immune=vaccines_immune,
        vaccines_other=vaccines_other,
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


def run_ode_fit(infections: pd.Series, ode_parameters: ODEParameters, progress_bar: bool) -> pd.DataFrame:
    beta_fit = []
    ode_parameter_list = tqdm.tqdm(list(ode_parameters), disable=not progress_bar)
    for location_id, location_params in ode_parameter_list:
        loc_beta_fit = run_loc_ode_fit(
            infections.loc[location_id],
            location_params
        )
        loc_beta_fit['location_id'] = location_id
        loc_beta_fit = loc_beta_fit.set_index(['location_id', 'date'])
        beta_fit.append(loc_beta_fit)
    return pd.concat(beta_fit).sort_index()


def run_loc_ode_fit(infections: pd.Series, ode_parameters: ODEParameters) -> pd.DataFrame:
    # Filter out early dates with few infections
    # to reduce noise in the past fit from leaking into the beta regression.
    infections = filter_to_epi_threshold(infections)
    ode_parameters = ode_parameters.reindex(infections.index)

    date = pd.Series(infections.index.values)
    t = (date - date.min()).dt.days.values
    obs = infections.values
    total_population = ode_parameters.population.iloc[0]

    initial_condition = np.zeros(len(past_system.COMPARTMENTS))
    initial_condition[past_system.s] = total_population - obs[0] - (obs[0] / 5) ** (1.0 / ode_parameters.alpha[0])
    initial_condition[past_system.e] = obs[0]
    initial_condition[past_system.i1] = (obs[0] / 5) ** (1.0 / ode_parameters.alpha[0])

    parameters = np.zeros((len(past_system.PARAMETERS), len(obs)))
    parameters[past_system.alpha] = ode_parameters.alpha.values
    parameters[past_system.sigma] = ode_parameters.sigma.values
    parameters[past_system.gamma1] = ode_parameters.gamma1.values
    parameters[past_system.gamma2] = ode_parameters.gamma2.values
    parameters[past_system.new_e] = obs
    parameters[past_system.m] = ode_parameters.vaccines_immune.values
    parameters[past_system.u] = ode_parameters.vaccines_other.values

    result = math.solve_ode(
        system=past_system.system,
        t=t,
        init_cond=initial_condition,
        params=parameters,
    )
    components = pd.DataFrame(
        data=result.T,
        columns=past_system.COMPARTMENTS,
    )
    components['date'] = date

    assert (components['S'] >= 0.0).all()

    susceptible = components.iloc[:, [past_system.s, past_system.s_u]].sum(axis=1)
    infectious = components.iloc[:, [past_system.i1, past_system.i2, past_system.i1_u, past_system.i2_u]].sum(axis=1)
    disease_density = susceptible * infectious**ode_parameters.alpha.values / total_population
    components['beta'] = (obs / disease_density)

    return components


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

import itertools
from typing import Dict, List, Union
import hashlib

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    solver,
)


def prepare_ode_fit_parameters(rates: pd.DataFrame,
                               epi_measures: pd.DataFrame,
                               rhos: pd.DataFrame,
                               vaccinations: pd.DataFrame,
                               etas: pd.DataFrame,
                               natural_waning_dist: pd.Series,
                               natural_waning_matrix: pd.DataFrame,
                               regression_params: Dict[str, Union[int, List[int]]],
                               draw_id: int) -> Parameters:
    past_index = rates.index
    sampled_params = sample_params(
        rates.index, regression_params,
        draw_id=draw_id,
    )
    sampled_params = pd.DataFrame(
        {k if 'kappa' in k else f'{k}_all_infection': v for k, v in sampled_params.items()})
    weights = []
    for measure in ['death', 'admission', 'case']:
        parameter = f'weight_all_{measure}'
        weights.append(
            pd.Series(sample_parameter(parameter, draw_id, 0., 1.), name=parameter, index=past_index)
        )
    weights = [w / sum(weights).rename(w.name) for w in weights]
    rhos = rhos.reindex(past_index, fill_value=0.)
    rhos.columns = [f'rho_{c}_infection' for c in rhos.columns]
    rhos['rho_none_infection'] = pd.Series(0., index=past_index, name='rho_none_infection')
    
    base_parameters = pd.concat([
        sampled_params,
        epi_measures.rename(columns=lambda x: f'count_all_{x[:-1]}'),
        pd.Series(-1, index=past_index, name='beta_all_infection'),
        rhos,
        *weights,
    ], axis=1)

    vaccinations = vaccinations.reindex(past_index, fill_value=0.)
    etas = etas.sort_index().reindex(past_index, fill_value=0.)

    phis = []
    for endpoint in ['infection', 'death', 'admission', 'case']:
        if endpoint == 'infection':
            w_base = pd.Series(0., index=natural_waning_dist.index)
        else:
            w_base = natural_waning_dist['infection']
        w_target = natural_waning_dist[endpoint]

        for from_variant, to_variant in itertools.product(VARIANT_NAMES, VARIANT_NAMES):
            cvi = natural_waning_matrix.loc[from_variant, to_variant]
            phi = 1 - (1 - cvi * w_target) / (1 - cvi * w_base)
            phis.append(phi.rename(f'{from_variant}_{to_variant}_{endpoint}'))
    phis = pd.concat(phis, axis=1)

    rates_map = {'ifr': 'death', 'ihr': 'admission', 'idr': 'case'}
    keep_cols = [f'{r}_{g}' for r, g in itertools.product(rates_map, RISK_GROUP_NAMES)]
    rates = rates.loc[:, keep_cols].rename(columns=lambda x: f"{rates_map[x.split('_')[0]]}_{x.split('_')[1]}")
    return Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        rates=rates,
        etas=etas,
        phis=phis,
    )


def sample_parameter(parameter: str, draw_id: int, lower: float, upper: float) -> float:
    key = f'{parameter}_{draw_id}'
    # 4294967295 == 2**32 - 1 which is the maximum allowable seed for a `numpy.random.RandomState`.
    seed = int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295
    random_state = np.random.RandomState(seed=seed)
    return random_state.uniform(lower, upper)


def sample_params(past_index: pd.Index,
                  param_dict: Dict,
                  draw_id: int) -> Dict[str, pd.Series]:
    sampled_params = {}
    for parameter in param_dict:
        param_spec = param_dict[parameter]
        if isinstance(param_spec, (int, float)):
            value = param_spec
        else:
            value = sample_parameter(parameter, draw_id, *param_spec)

        sampled_params[parameter] = pd.Series(
            value,
            index=past_index,
            name=parameter,
        )

    return sampled_params


def make_initial_condition(parameters: Parameters, full_rates: pd.DataFrame, population: pd.DataFrame):
    base_params = parameters.base_parameters
    
    crude_infections = get_crude_infections(base_params, full_rates, population)    
    new_e_start = crude_infections.reset_index(level='date').groupby('location_id').first()
    start_date, new_e_start = new_e_start['date'], new_e_start[list(RISK_GROUP_NAMES)]
    end_date = crude_infections.reset_index(level='date').groupby('location_id').date.last() + pd.Timedelta(days=1)    

    # Alpha is time-invariant
    alpha = base_params.alpha_all_infection.groupby('location_id').first()
    compartments = [f'{compartment}_{risk_group}'
                    for risk_group, compartment
                    in itertools.product(RISK_GROUP_NAMES, COMPARTMENTS_NAMES + TRACKING_COMPARTMENTS_NAMES)]
    initial_condition = []
    for location_id, loc_start_date in start_date.iteritems():
        loc_end_date = end_date.loc[location_id]
        loc_initial_condition = pd.DataFrame(0., columns=compartments, index=full_rates.loc[location_id].index)
        for risk_group in RISK_GROUP_NAMES:
            pop = population.loc[location_id, risk_group]
            new_e = new_e_start.loc[location_id, risk_group]
            suffix = f'_unvaccinated_{risk_group}'
            # Backfill everyone susceptible
            loc_initial_condition.loc[:loc_start_date, f'S_none{suffix}'] = pop
            # Set initial condition on start date
            infectious = (new_e / 5) ** (1 / alpha.loc[location_id])
            loc_initial_condition.loc[loc_start_date, f'S_none{suffix}'] = pop - new_e - infectious
            loc_initial_condition.loc[loc_start_date, f'E_ancestral{suffix}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'NewE_ancestral{suffix}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'NewE_ancestral_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'I_ancestral{suffix}'] = infectious
            for variant in VARIANT_NAMES:
                loc_initial_condition.loc[:loc_start_date, f'EffectiveSusceptible_{variant}{suffix}'] = pop
        loc_initial_condition.loc[loc_end_date:, :] = np.nan
        loc_initial_condition['location_id'] = location_id
        loc_initial_condition = loc_initial_condition.set_index('location_id', append=True).reorder_levels(['location_id', 'date'])
        initial_condition.append(loc_initial_condition)
    initial_condition = pd.concat(initial_condition)

    return initial_condition


def get_crude_infections(base_params, rates, population, threshold = 50):
    crude_infections = pd.DataFrame(index=rates.index)
    for risk_group in RISK_GROUP_NAMES:
        risk_infections = pd.DataFrame(index=rates.index)
        for measure, rate in [('death', 'ifr'), ('admission', 'ihr'), ('case', 'idr')]:
            infections = (base_params[f'weight_all_{measure}']
                          * base_params[f'count_all_{measure}'] / rates[rate]
                          * population[risk_group] / population.sum(axis=1))
               
            risk_infections[measure] = infections
        not_null = risk_infections.notnull().any(axis=1)
        risk_infections = risk_infections.sum(axis=1).loc[not_null].rename(risk_group)
        crude_infections[risk_group] = risk_infections
    
    crude_infections = crude_infections.loc[crude_infections.sum(axis=1) > threshold]    
    return crude_infections


def run_ode_fit(initial_condition: pd.DataFrame, ode_parameters: Parameters, progress_bar: bool):
    full_compartments, chis = solver.run_ode_model(
        initial_condition,
        *ode_parameters.to_dict(),
        forecast=False,
        num_cores=5,
        progress_bar=progress_bar,
    )
    # Set all the forecast stuff to nan
    full_compartments.loc[full_compartments.sum(axis=1) == 0., :] = np.nan
    # All the same so mean just collapses.
    beta = full_compartments.filter(like='beta_none_all').mean(axis=1).groupby('location_id').diff().rename('beta')
    # Don't want to break the log.
    beta[beta == 0.] = np.nan
    return beta, chis, full_compartments


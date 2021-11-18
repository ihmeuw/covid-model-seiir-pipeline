import itertools
from typing import Dict, List, Tuple
import hashlib

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


def prepare_ode_fit_parameters(rates: pd.DataFrame,
                               epi_measures: pd.DataFrame,
                               rhos: pd.DataFrame,
                               vaccinations: pd.DataFrame,
                               etas: pd.DataFrame,
                               natural_waning_dist: pd.Series,
                               natural_waning_matrix: pd.DataFrame,
                               sampled_params: Dict[str, pd.Series]) -> Parameters:
    past_index = rates.index

    sampled_params = pd.DataFrame(
        {k if 'kappa' in k or 'zeta' in k else f'{k}_all': v for k, v in sampled_params.items()})
    rhos = rhos.reindex(past_index, fill_value=0.)
    rhos.columns = [f'rho_{c}' for c in rhos.columns]
    rhos['rho_none'] = pd.Series(0., index=past_index, name='rho_none')

    base_parameters = pd.concat([
        sampled_params,
        epi_measures.rename(columns=lambda x: f'{x}_all'),
        pd.Series(-1, index=past_index, name='beta_all'),
        rhos,
    ], axis=1)

    vaccinations = vaccinations.reindex(past_index, fill_value=0.)
    etas = etas.set_index('endpoint', append=True).reorder_levels(['endpoint', 'location_id', 'date']).sort_index()
    etas = {f'eta_{endpoint}': etas.loc[endpoint] for endpoint in ['infection', 'death', 'admission', 'case']}
    natural_waning_dist = {f'natural_waning_{endpoint}': natural_waning_dist.loc[endpoint] for endpoint in
                           ['infection', 'death', 'admission', 'case']}

    return Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        rates=rates,
        **etas,
        **natural_waning_dist,
        phi=natural_waning_matrix,
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
            initial_condition.loc[(location_id, loc_start_date), f'NewE_ancestral_all_{risk_group}'] = new_e
            initial_condition.loc[(location_id, loc_start_date), f'I_ancestral{suffix}'] = infectious
            for variant in VARIANT_NAMES:
                initial_condition.loc[
                    pd.IndexSlice[location_id, :loc_start_date], f'EffectiveSusceptible_{variant}{suffix}'] = pop

    return initial_condition



def run_ode_fit(initial_condition: pd.DataFrame, ode_parameters: Parameters, progress_bar: bool):
    full_compartments, chis = solver.run_ode_model(
        initial_condition,
        *ode_parameters.to_dfs(),
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


from typing import Dict

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import sample_idr_parameters
from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters


def rescale_kappas(sampled_ode_params: Dict,
                   compartments: pd.DataFrame,
                   rates_parameters: RatesParameters,
                   hierarchy: pd.DataFrame,
                   draw_id: int):
    hierarchy = hierarchy.loc[hierarchy['most_detailed'] == 1]
    us_locations = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '102' in x.split(',')),
                                 'location_id'].to_list()
    spain_locations = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '92' in x.split(',')),
                                    'location_id'].to_list()
    india_locations = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '163' in x.split(',')),
                                    'location_id'].to_list()

    delta_infections = compartments.filter(like='Infection_all_delta_all').sum(axis=1).groupby('location_id').max()
    delta_cases = compartments.filter(like='Case_all_delta_all').sum(axis=1).groupby('location_id').max()
    all_infections = compartments.filter(like='Infection_all_all_all').sum(axis=1).groupby('location_id').max()
    all_cases = compartments.filter(like='Case_all_all_all').sum(axis=1).groupby('location_id').max()
    max_idr = 0.9

    idr_parameters = sample_idr_parameters(rates_parameters, draw_id)
    p_symptomatic_pre_omicron = 1 - idr_parameters['p_asymptomatic_pre_omicron']
    p_symptomatic_post_omicron = 1 - idr_parameters['p_asymptomatic_post_omicron']
    minimum_asymptomatic_idr_fraction = idr_parameters['minimum_asymptomatic_idr_fraction']
    maximum_asymptomatic_idr = idr_parameters['maximum_asymptomatic_idr']

    idr_scaling_factors = [
        (   55,  3.0),  # Slovenia
        (   60,  3.0),  # Lithuania
        (43860,  3.0),  # Manitoba
        (  531,  3.0),  # District of Columbia
        (   74,  3.0),  # Andorra
        (   83,  5.0),  # Iceland
        [  186,  5.0],  # Seychelles
        (  169,  5.0),  # Central African Republic
        (  181,  5.0),  # Madagascar
    ]
    # IDR = p_s * IDR_s + p_a * IDR_a
    # IDR_a = (IDR - IDR_s * p_s) / p_a
    # min_a_frac * IDR <= IDR_a <= max_a
    delta_idr = delta_cases / delta_infections
    delta_idr = delta_idr.fillna(all_cases / all_infections)
    capped_delta_idr = np.minimum(delta_idr, max_idr)
    idr_asymptomatic = (capped_delta_idr - max_idr * p_symptomatic_pre_omicron) / (1 - p_symptomatic_pre_omicron)
    idr_asymptomatic = np.maximum(idr_asymptomatic, capped_delta_idr * minimum_asymptomatic_idr_fraction)
    idr_symptomatic = (capped_delta_idr - idr_asymptomatic * (1 - p_symptomatic_pre_omicron)) / p_symptomatic_pre_omicron
    idr_asymptomatic = np.minimum(idr_asymptomatic, maximum_asymptomatic_idr)
    omicron_idr = p_symptomatic_post_omicron * idr_symptomatic + (1 - p_symptomatic_post_omicron) * idr_asymptomatic
    for location_id, idr_scaling_factor in idr_scaling_factors:
        omicron_idr.loc[location_id] *= idr_scaling_factor
    sampled_ode_params['kappa_omicron_case'] = (omicron_idr / delta_idr).rename('kappa_omicron_case')

    ihr_scaling_factors = [
        (43860,  3.0),  # Manitoba
        (  531,  3.0),  # District of Columbia
    ]
    kappa_omicron_admission = pd.Series(
        sampled_ode_params['kappa_omicron_admission'],
        index=omicron_idr.index,
        name='kappa_omicron_admission'
    )
    for location_id, ihr_scaling_factor in ihr_scaling_factors:
        kappa_omicron_admission.loc[location_id] *= ihr_scaling_factor
    sampled_ode_params['kappa_omicron_admission'] = kappa_omicron_admission

    ifr_scaling_factors = [
        (   34,  3.0),  # Azerbaijan
        (   44,  3.0),  # Bosnia and Herzegovina
        (   45,  3.0),  # Bulgaria
        (   49,  3.0),  # North Macedonia
        (   55,  3.0),  # Slovenia
        (   60,  3.0),  # Lithuania
        (43860,  3.0),  # Manitoba
        (43862,  5.0),  # Newfoundland and Labrador
        (   80,  2.0),  # France
        (   82,  2.0),  # Greece
        (   85,  2.0),  # Israel
        (  118,  3.0),  # Suriname
        (  119,  3.0),  # Trinidad and Tobago
        (  169,  5.0),  # Central African Republic
        (  181, 10.0),  # Madagascar
    ]
    ifr_scaling_factors += [(loc_id, 2.0) for loc_id in us_locations]  # United States of America
    ifr_scaling_factors += [(loc_id, 2.0) for loc_id in spain_locations]  # Spain
    ifr_scaling_factors += [(loc_id, 2.0) for loc_id in india_locations]  # India
    kappa_omicron_death = pd.Series(
        sampled_ode_params['kappa_omicron_death'],
        index=omicron_idr.index,
        name='kappa_omicron_death'
    )
    for location_id, ifr_scaling_factor in ifr_scaling_factors:
        kappa_omicron_death.loc[location_id] *= ifr_scaling_factor
    sampled_ode_params['kappa_omicron_death'] = kappa_omicron_death
    return sampled_ode_params

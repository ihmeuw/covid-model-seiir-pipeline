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
        (   37,  0.4),  # Kyrgyzstan
        (   41,  0.6),  # Uzbekistan
        (   55,  3.0),  # Slovenia
        (   60,  3.0),  # Lithuania
        (43860,  3.0),  # Manitoba
        (  531,  3.0),  # District of Columbia
        (   74,  3.0),  # Andorra
        (   83,  5.0),  # Iceland
        (60358,  3.0),  # Aragon
        (60365,  3.0),  # Asturias
        (60363,  3.0),  # Balearic Islands
        (60364,  3.0),  # Canary Islands
        (60373,  3.0),  # Melilla
        (  109,  0.2),  # Cuba
        (  114,  0.2),  # Hati
        (  139,  0.2),  # Algeria
        (  152,  0.4),  # Saudi Arabia
        (  522,  0.4),  # Sudan
        (  157,  0.2),  # Yemen
        [  186,  5.0],  # Seychelles
        (  169,  3.0),  # Central African Republic
        (  172,  0.4),  # Equatorial Guinea
        (  175,  0.2),  # Burundi
        (  177,  0.4),  # Djibouti
        (  178,  0.4),  # Eritrea
        (  181,  5.0),  # Madagascar
        (  200,  0.2),  # Benin
        (  202,  0.2),  # Cameroon
        (  204,  0.2),  # Chad
        (  206,  0.4),  # Gambia
        (  209,  0.4),  # Guinea-Bissau
        (  213,  0.4),  # Niger

        ## ## ## ## ## ## INDIA ## ## ## ## ## ##
        ( 4841,  0.2),  # Andhra Pradesh
        ( 4842,  0.2),  # Arunachal Pradesh
        ( 4843,  0.6),  # Assam
        ( 4844,  0.4),  # Bihar
        ( 4846,  0.4),  # Chhattisgarh
        # Delhi - no change
        ( 4850,  0.4),  # Goa
        ( 4851,  0.4),  # Gujarat
        ( 4852,  0.8),  # Haryana
        ( 4853,  0.6),  # Himachal Pradesh
        ( 4854,  0.4),  # Jammu & Kashmir and Ladakh
        ( 4855,  0.6),  # Jharkhand
        ( 4856,  0.6),  # Karnataka
        # Kerala - no change
        ( 4859,  0.4),  # Madhya Pradesh
        ( 4860,  0.6),  # Maharashtra
        ( 4861,  0.2),  # Manipur
        ( 4862,  0.2),  # Meghalaya
        # Mizoram - no change
        ( 4864,  0.2),  # Nagaland
        ( 4865,  0.4),  # Odisha
        ( 4867,  0.6),  # Punjab
        ( 4868,  0.6),  # Rajasthan
        ( 4869,  0.4),  # Sikkim
        ( 4870,  0.6),  # Tamil Nadu
        ( 4871,  0.2),  # Telangana
        ( 4872,  0.4),  # Tripura
        ( 4873,  0.4),  # Uttar Pradesh
        ( 4874,  0.8),  # Uttarakhand
        ( 4875,  0.8),  # West Bengal
        ## ## ## ## ## ## ## ## ## ## ## ## ## ##
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
        (60358,  3.0),  # Aragon
        (60365,  3.0),  # Asturias
        (60364,  3.0),  # Canary Islands
        (60373,  3.0),  # Melilla
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
        (   22,  5.0),  # Fiji
        (  169,  3.0),  # Central African Republic
        (  175,  0.2),  # Burundi
        (  177,  0.4),  # Djibouti
        (  178,  0.4),  # Eritrea
        (  181,  5.0),  # Madagascar
    ]
    ifr_scaling_factors += [(loc_id, 2.0) for loc_id in us_locations]  # United States of America
    ifr_scaling_factors += [(loc_id, 2.0) for loc_id in spain_locations]  # Spain
    kappa_omicron_death = pd.Series(
        sampled_ode_params['kappa_omicron_death'],
        index=omicron_idr.index,
        name='kappa_omicron_death'
    )
    for location_id, ifr_scaling_factor in ifr_scaling_factors:
        kappa_omicron_death.loc[location_id] *= ifr_scaling_factor
    sampled_ode_params['kappa_omicron_death'] = kappa_omicron_death
    return sampled_ode_params

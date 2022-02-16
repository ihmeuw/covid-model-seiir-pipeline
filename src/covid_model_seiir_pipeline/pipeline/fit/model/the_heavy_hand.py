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
    # us_locations = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '102' in x.split(',')),
    #                              'location_id'].to_list()
    # spain_locations = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '92' in x.split(',')),
    #                                 'location_id'].to_list()
    ita_locations = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '86' in x.split(',')),
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
        (   35,  3.0),  # Georgia
        (   38,  3.0),  # Mongolia
        (   55,  3.0),  # Slovenia
        (   59,  3.0),  # Latvia
        (   60,  3.0),  # Lithuania
        (   62,  3.0),  # Russian Federation
        (   63,  3.0),  # Ukraine
        (  349,  5.0),  # Greenland
        (   83,  3.0),  # Iceland
        (   90,  3.0),  # Norway
        (   91,  3.0),  # Portugal
        (  396,  3.0),  # San Marino
        (  121,  3.0),  # Bolivia
        (  118,  3.0),  # Suriname
        ( 4757,  3.0),  # Espirito Santo
        (  140,  3.0),  # Bahrain
        (  147,  3.0),  # Libya
        (  151,  3.0),  # Qatar
        ( 4851,  3.0),  # Gujarat
        ( 4869,  3.0),  # Sikkim
        (  351,  5.0),  # Guam
        (   23, 10.0),  # Kiribati
        (  376,  3.0),  # Northern Mariana Islands
        (   28, 10.0),  # Solomon Islands
        (   14,  3.0),  # Maldives
        (  186,  3.0),  # Seychelles
        (  168,  3.0),  # Angola
        (  198,  3.0),  # Zimbabwe
        (  211,  3.0),  # Mali
        (  212,  3.0),  # Mauritania
        (  215,  3.0),  # Sao Tome and Principe
        (  216,  3.0),  # Senegal
    ]
    idr_scaling_factors += [(loc_id, 3.0) for loc_id in ita_locations]  # Italy
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
        (   47,  3.0),  # Czechia
        (43860,  3.0),  # Manitoba
        ( 4655,  3.0),  # Hidalgo
        ( 4669,  3.0),  # Tabasco
        ( 4673,  3.0),  # Yucatan
        (  151,  3.0),  # Qatar
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
        (   33,  3.0),  # Armenia
        (   34,  3.0),  # Azerbaijan
        (   37,  3.0),  # Kyrgyzstan
        (   41,  3.0),  # Uzbekistan
        (   43,  3.0),  # Albania
        (   44,  3.0),  # Bosnia and Herzegovina
        (   45,  3.0),  # Bulgaria
        (   45,  3.0),  # Croatia
        (   47,  3.0),  # Czechia
        (   50,  3.0),  # Montenegro
        (   49,  3.0),  # North Macedonia
        (   51,  3.0),  # Poland
        (   53,  3.0),  # Serbia
        (   57,  3.0),  # Belarus
        (  349,  5.0),  # Greenland
        (  122,  3.0),  # Ecuador
        (  113,  3.0),  # Guyana
        (  118,  3.0),  # Suriname
        (  129,  3.0),  # Honduras
        ( 4644,  3.0),  # Baja California
        ( 4645,  3.0),  # Baja California Sur
        ( 4650,  3.0),  # Chihuahua
        ( 4647,  3.0),  # Coahuila
        ( 4652,  3.0),  # Durango
        ( 4653,  3.0),  # Guanajuato
        ( 4655,  3.0),  # Hidalgo
        ( 4651,  3.0),  # Mexico City
        ( 4661,  3.0),  # Nuevo Leon
        ( 4665,  3.0),  # Quintana Roo
        ( 4759,  3.0),  # Maranhao
        ( 4762,  3.0),  # Mato Grosso
        ( 4761,  3.0),  # Mato Grosso do Sul
        ( 4775,  3.0),  # Sao Paulo
        (  136,  3.0),  # Paraguay
        (  143,  3.0),  # Iraq
        (  144,  3.0),  # Jordan
        (  146,  3.0),  # Lebanon
        (  149,  3.0),  # Palestine
        (  522,  3.0),  # Sudan
        (  154,  3.0),  # Tunisia
        (  155,  3.0),  # Turkey
        ( 4852,  3.0),  # Haryana
        ( 4862,  3.0),  # Meghalaya
        ( 4865,  3.0),  # Odisha
        ( 4872,  3.0),  # Tripura
        ( 4875,  3.0),  # West Bengal
        (  169,  3.0),  # Central African Republic
        (  171,  3.0),  # Democratic Republic of the Congo
        (  173,  3.0),  # Gabon
        (  179,  3.0),  # Ethiopia
        (  180,  3.0),  # Kenya
        (  181,  3.0),  # Madagascar
        (  182,  3.0),  # Malawi
        (  184,  3.0),  # Mozambique
        (  190,  3.0),  # Uganda
        (  201,  3.0),  # Burkina Faso
        (  202,  3.0),  # Cameroon
        (  205,  3.0),  # Cote d'Ivoire
        (  206,  3.0),  # Gambia
        (  208,  3.0),  # Guinea
        (  209,  3.0),  # Guinea-Bissau
        (  211,  3.0),  # Mali
        (  215,  3.0),  # Sao Tome and Principe
    ]
    kappa_omicron_death = pd.Series(
        sampled_ode_params['kappa_omicron_death'],
        index=omicron_idr.index,
        name='kappa_omicron_death'
    )
    for location_id, ifr_scaling_factor in ifr_scaling_factors:
        kappa_omicron_death.loc[location_id] *= ifr_scaling_factor
    sampled_ode_params['kappa_omicron_death'] = kappa_omicron_death
    return sampled_ode_params

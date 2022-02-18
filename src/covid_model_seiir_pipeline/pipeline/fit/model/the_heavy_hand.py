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
        (   35,  2.0),  # Georgia
        (   47,  2.0),  # Czechia
        (   50, 10.0),  # Montenegro
        (   51,  2.0),  # Poland
        (   55,  2.0),  # Slovenia
        (   59,  2.0),  # Latvia
        (   60,  2.0),  # Lithuania
        (   62,  3.0),  # Russian Federation
        (43860,  3.0),  # Manitoba
        (  349,  5.0),  # Greenland
        (  531,  2.0),  # District of Columbia
        ( 3539,  2.0),  # Spokane County
        (   74,  2.0),  # Andorra
        (   78,  2.0),  # Denmark
        (   90,  2.0),  # Norway
        (   91,  2.0),  # Portugal
        (  396,  3.0),  # San Marino
        ( 4636,  2.0),  # Wales
        (  118,  3.0),  # Suriname
        ( 4654,  0.5),  # Guerrero
        ( 4658,  0.5),  # Michoacan de Ocampo
        ( 4659,  0.5),  # Morelos
        ( 4662,  0.5),  # Oaxaca
        ( 4664,  0.5),  # Queretaro
        (  133,  0.2),  # Venezuela
        ( 4750,  0.5),  # Acre
        ( 4752,  0.5),  # Amazonas
        ( 4754,  0.5),  # Bahia
        ( 4756,  0.5),  # Distrito Federal
        ( 4755,  0.2),  # Ceara
        ( 4757,  2.0),  # Espirito Santo
        ( 4758,  0.5),  # Goias
        ( 4759,  0.2),  # Maranhao
        ( 4762,  0.5),  # Mato Grosso
        ( 4763,  0.5),  # Para
        ( 4764,  0.2),  # Paraiba
        ( 4766,  0.5),  # Pernambuco
        ( 4767,  0.2),  # Piaui
        ( 4769,  0.2),  # Rio Grande do Norte
        ( 4770,  0.5),  # Rondonia
        ( 4773,  0.5),  # Santa Catarina
        ( 4774,  0.2),  # Sergipe
        ( 4776,  0.5),  # Tocantins
        (  160,  0.2),  # Afghanistan
        (  140,  2.0),  # Bahrain
        (  139,  0.5),  # Algeria
        (  141,  0.5),  # Egypt
        (  142,  0.2),  # Iran
        (  143,  0.5),  # Iraq
        (  144,  2.0),  # Jordan
        (  151,  2.0),  # Qatar
        (  155,  2.0),  # Turkey
        (  157,  0.2),  # Yemen
        ( 4849,  5.0),  # Delhi
        ( 4851,  2.0),  # Gujarat
        ( 4852,  2.0),  # Haryana
        ( 4863,  5.0),  # Mizoram
        ( 4869,  2.0),  # Sikkim
        ( 4874,  2.0),  # Uttarakhand
        (53616,  0.2),  # Balochistan
        (53617,  0.5),  # Gilgit-Baltistan
        (53618,  0.5),  # Islamabad Capital Territory
        (53619,  0.5),  # Khyber Pakhtunkhwa
        (53620,  0.2),  # Punjab
        (  351,  5.0),  # Guam
        (   23, 10.0),  # Kiribati
        (  376,  3.0),  # Northern Mariana Islands
        (   28,  5.0),  # Solomon Islands
        (   10,  0.5),  # Cambodia
        (   13,  0.5),  # Malaysia
        (   15,  0.5),  # Myanmar
        (  186,  5.0),  # Seychelles
        (   17,  0.5),  # Sri Lanka
        (   19,  0.5),  # Timor-Leste
        (   20,  0.5),  # Viet Nam
        (  168,  5.0),  # Angola
        (  173,  0.2),  # Gabon
        (  182,  3.0),  # Malawi
        (  204,  0.2),  # Chad
        (  211,  3.0),  # Mali
        (  212,  2.0),  # Mauritania
        (  216,  2.0),  # Senegal
        (  217,  0.5),  # Sierra Leone
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
        (   47,  2.0),  # Czechia
        (  567,  2.0),  # Utah
        ( 3539,  2.0),  # Spokane County
        (   78,  2.0),  # Denmark
        ( 4644,  2.0),  # Baja California
        ( 4645,  2.0),  # Baja California Sur
        ( 4649,  0.2),  # Chiapas
        ( 4647,  3.0),  # Coahuila
        ( 4652,  3.0),  # Durango
        ( 4653,  3.0),  # Guanajuato
        ( 4654,  0.5),  # Guerrero
        ( 4658,  0.5),  # Michoacan de Ocampo
        ( 4662,  0.5),  # Oaxaca
        ( 4664,  0.5),  # Queretaro
        ( 4672,  0.5),  # Veracruz de Ignacio de la Llave
        ( 4751,  0.2),  # Alagoas
        ( 4753,  0.5),  # Amapa
        ( 4758,  0.5),  # Goias
        ( 4763,  0.5),  # Para
        ( 4764,  0.5),  # Paraiba
        ( 4766,  0.2),  # Pernambuco
        ( 4767,  0.2),  # Piaui
        ( 4769,  0.2),  # Rio Grande do Norte
        ( 4770,  0.5),  # Rondonia
        ( 4773,  0.5),  # Santa Catarina
        ( 4774,  0.5),  # Sergipe
        ( 4776,  0.5),  # Tocantins
        (  151,  2.0),  # Qatar
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
        (   34,  5.0),  # Azerbaijan
        (   43,  3.0),  # Albania
        (   44,  3.0),  # Bosnia and Herzegovina
        (   45, 10.0),  # Croatia
        (   47,  2.0),  # Czechia
        (   50,  5.0),  # Montenegro
        (   49,  3.0),  # North Macedonia
        (   51,  2.0),  # Poland
        (   53,  2.0),  # Serbia
        (   57,  3.0),  # Belarus
        (   59,  2.0),  # Latvia
        (   60,  2.0),  # Lithuania
        (   62,  3.0),  # Russian Federation
        (  349,  5.0),  # Greenland
        (  535,  2.0),  # Idaho
        ( 3539,  2.0),  # Spokane County
        (  573,  2.0),  # Wyoming
        (   78,  2.0),  # Denmark
        (35498,  2.0),  # Provincia autonoma di Bolzano
        (   90,  2.0),  # Norway
        (  122,  2.0),  # Ecuador
        (  113,  3.0),  # Guyana
        (  118,  3.0),  # Suriname
        ( 4644,  2.0),  # Baja California
        ( 4645,  2.0),  # Baja California Sur
        ( 4649,  0.2),  # Chiapas
        ( 4647,  2.0),  # Coahuila
        ( 4652,  2.0),  # Durango
        ( 4653,  2.0),  # Guanajuato
        ( 4654,  0.5),  # Guerrero
        ( 4651,  2.0),  # Mexico City
        ( 4661,  2.0),  # Nuevo Leon
        ( 4665,  2.0),  # Quintana Roo
        ( 4670,  2.0),  # Tamaulipas
        ( 4672,  0.5),  # Veracruz de Ignacio de la Llave
        ( 4674,  2.0),  # Zacatecas
        (  133,  0.2),  # Venezuela
        ( 4751,  0.2),  # Alagoas
        ( 4754,  0.5),  # Bahia
        ( 4756,  0.5),  # Distrito Federal
        ( 4758,  0.5),  # Goias
        ( 4760,  0.5),  # Minas Gerais
        ( 4763,  0.5),  # Para
        ( 4764,  0.5),  # Paraiba
        ( 4765,  0.5),  # Parana
        ( 4766,  0.5),  # Pernambuco
        ( 4767,  0.2),  # Piaui
        ( 4769,  0.5),  # Rio Grande do Norte
        ( 4768,  0.2),  # Rio de Janeiro
        ( 4770,  0.5),  # Rondonia
        ( 4773,  0.5),  # Santa Catarina
        ( 4774,  0.5),  # Sergipe
        ( 4776,  0.5),  # Tocantins
        (  136,  3.0),  # Paraguay
        (  139,  0.5),  # Algeria
        (  143,  0.5),  # Iraq
        (  144,  2.0),  # Jordan
        (  146,  3.0),  # Lebanon
        (  147,  2.0),  # Libya
        (  149,  3.0),  # Palestine
        (  152,  0.5),  # Saudi Arabia
        (  522,  2.0),  # Sudan
        (  154,  3.0),  # Tunisia
        (  155,  2.0),  # Turkey
        ( 4849,  5.0),  # Delhi
        ( 4852,  2.0),  # Haryana
        ( 4861,  2.0),  # Manipur
        ( 4862,  2.0),  # Meghalaya
        ( 4863,  5.0),  # Mizoram
        ( 4865,  2.0),  # Odisha
        ( 4871,  0.5),  # Telangana
        ( 4872,  3.0),  # Tripura
        ( 4874,  2.0),  # Uttarakhand
        ( 4875,  2.0),  # West Bengal
        (  168,  5.0),  # Angola
        (  171,  2.0),  # Democratic Republic of the Congo
        (  172,  3.0),  # Equatorial Guinea
        (  176,  0.2),  # Comoros
        (  179,  3.0),  # Ethiopia
        (  180,  3.0),  # Kenya
        (  181,  2.0),  # Madagascar
        (  182,  3.0),  # Malawi
        (  184,  3.0),  # Mozambique
        (  187,  0.2),  # Somalia
        (  190,  2.0),  # Uganda
        (  194,  3.0),  # Lesotho
        (  195,  2.0),  # Namibia
        (  198,  3.0),  # Zimbabwe
        (  201,  3.0),  # Burkina Faso
        (  205,  3.0),  # Cote d'Ivoire
        (  206,  3.0),  # Gambia
        (  208,  3.0),  # Guinea
        (  211,  3.0),  # Mali
        (  212,  2.0),  # Mauritania
        (  215,  5.0),  # Sao Tome and Principe
        (  216,  2.0),  # Senegal
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

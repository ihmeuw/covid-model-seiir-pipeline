from typing import Dict

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import sample_idr_parameters
from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters


def rescale_kappas(measure: str,
                   sampled_ode_params: Dict,
                   compartments: pd.DataFrame,
                   rates_parameters: RatesParameters,
                   hierarchy: pd.DataFrame,
                   draw_id: int):
    test_scalar = rates_parameters.test_scalar

    if measure == 'case':
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

        idr_scaling_factors = {
            'ancestral': [
                (   37,  5.0),  # Kyrgyzstan
                (  168,  1.5),  # Angola
                (  214,  0.5),  # Nigeria
            ],
            'alpha': [
                (   33,  3.0),  # Armenia
                (   37,  5.0),  # Kyrgyzstan
                (   61,  3.0),  # Moldova
                (  121,  3.0),  # Bolivia
                (  168, 50.0),  # Angola
            ],
            'gamma': [
                (  121,  3.0),  # Bolivia
                ( 4757,  3.0),  # Espirito Santo
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
            ],
            'delta': [
                (   33,  3.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   37,  5.0),  # Kyrgyzstan
                (   38,  3.0),  # Mongolia
                (   43,  3.0),  # Albania
                (   50,  3.0),  # Montenegro
                (   61,  3.0),  # Moldova
                (   62,  2.0),  # Russia
                (  121,  3.0),  # Bolivia
                ( 4644,  2.0),  # Baja California
                ( 4653,  2.0),  # Guanajuato
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  2.0),  # Tabasco
                ( 4673,  3.0),  # Yucatan
                ( 4757,  3.0),  # Espirito Santo
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
                (  147,  2.0),  # Libya
                (  161,  2.0),  # Bangladesh
                ( 4842,  2.0),  # Arunachal Pradesh
                ( 4849,  2.0),  # Delhi
                ( 4861,  3.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  8.0),  # Mizoram
                ( 4845,  3.0),  # Chandigarh
                ( 4858,  4.0),  # Lakshadweep
                ( 4866,  5.0),  # Puducherry
                ( 4869,  2.0),  # Sikkim
                (53619,  2.0),  # Khyber Pakhtunkhwa
                (  186,  3.0),  # Seychelles
                (  168, 30.0),  # Angola
                (  169,  2.0),  # Central African Republic
                (  171,  2.0),  # DRC
                (  181,  2.0),  # Madagascar
                (  214,  0.2),  # Nigeria
            ],
            'omicron': [
                (   33,  2.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   35,  2.5),  # Georgia
                (   37,  5.0),  # Kyrgyzstan
                (   38,  3.0),  # Mongolia
                (   43,  3.0),  # Albania
                (   44,  0.8),  # Bosnia and Herzegovina
                (   50,  3.0),  # Montenegro
                (   49,  0.8),  # North Macedonia
                (   51,  2.0),  # Poland
                (   58,  2.0),  # Estonia
                (   59,  2.0),  # Latvia
                (   60,  3.0),  # Lithuania
                (   61,  5.0),  # Moldova
                (   62,  5.0),  # Russia
                (43860,  2.0),  # Manitoba
                (  531,  2.0),  # District of Columbia
                (  536,  1.5),  # Illinois
                (   74,  1.2),  # Andorra
                (   76,  1.2),  # Belgium
                (   78,  2.0),  # Denmark
                (60377,  1.5),  # Baden Wurttemberg
                (60378,  2.0),  # Bavaria
                (60379,  1.5),  # Berlin
                (60380,  1.5),  # Brandenberg
                (60381,  1.5),  # Bremen
                (60382,  1.5),  # Hamburg
                (60386,  1.5),  # North Rhine-Westphalia
                (   83,  3.0),  # Iceland
                (  396,  2.0),  # San Marino
                (60358,  1.5),  # Aragon
                (60365,  1.5),  # Asturias
                (60363,  2.0),  # Balearic Islands
                (60367,  2.0),  # Canary Islands
                (60370,  2.0),  # Navarre
                (  121,  3.0),  # Bolivia
                (  122,  0.8),  # Ecuador
                ( 4644,  2.0),  # Baja California
                ( 4647,  1.5),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4665,  2.0),  # Quintana Roo
                ( 4667,  2.0),  # Sinaloa
                ( 4669,  3.0),  # Tabasco
                ( 4673,  4.0),  # Yucatan
                ( 4756,  1.2),  # Distrito Federal
                ( 4757,  5.0),  # Espirito Santo
                ( 4758,  0.8),  # Goias
                ( 4759,  0.8),  # Maranhao
                ( 4764,  3.0),  # Paraiba
                ( 4765,  0.8),  # Parana
                ( 4770,  0.8),  # Rondonia
                ( 4771,  2.0),  # Roraima
                (  140,  3.0),  # Bahrain
                (  144,  3.0),  # Jordan
                (  147,  2.0),  # Libya
                (  155,  2.0),  # Turkey
                (  161,  2.0),  # Bangladesh
                ( 4842,  2.0),  # Arunachal Pradesh
                ( 4849,  3.0),  # Delhi
                ( 4851,  1.5),  # Gujarat
                ( 4861,  3.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863, 40.0),  # Mizoram
                ( 4845,  5.0),  # Chandigarh
                ( 4858,  4.0),  # Lakshadweep
                ( 4866, 10.0),  # Puducherry
                ( 4868,  0.5),  # Rajasthan
                ( 4869,  3.0),  # Sikkim
                (53617,  0.8),  # Gilgit-Baltistan
                (53619,  2.0),  # Khyber Pakhtunkhwa
                (  351,  5.0),  # Guam
                (   14,  1.5),  # Maldives
                (  186,  3.0),  # Seychelles
                (  168, 30.0),  # Angola
                (  169,  1.5),  # Central African Republic
                (  171,  5.0),  # DRC
                (  181,  2.0),  # Madagascar
                (  187,  0.8),  # Somalia
                (  198,  1.2),  # Zimbabwe
                (  203,  1.5),  # Cabo Verde
                (  204,  0.5),  # Chad
                (  202,  0.8),  # Cameroon
                (  211,  2.0),  # Mali
                (  214,  0.2),  # Nigeria
            ]
        }
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

        if rates_parameters.heavy_hand_fixes:
            for variant, scaling_factors in idr_scaling_factors.items():
                kappa = pd.Series(
                    sampled_ode_params[f'kappa_{variant}_case'],
                    index=compartments.reset_index().location_id.unique(),
                    name=f'kappa_{variant}_case'
                )
                for location_id, scaling_factor in scaling_factors:
                    kappa.loc[location_id] *= scaling_factor
                    if variant == 'delta':
                        delta_idr.loc[location_id] *= scaling_factor
                    if variant == 'omicron':
                        omicron_idr.loc[location_id] *= scaling_factor
                if variant == 'omicron':
                    sampled_ode_params['kappa_omicron_case'] = ((omicron_idr / delta_idr)
                                                                .rename('kappa_omicron_case'))
                else:
                    sampled_ode_params[f'kappa_{variant}_case'] = kappa

    if measure == 'admission':
        ihr_scaling_factors = {
            'gamma': [
                ( 4752,  1.5),  # Amazonas
                ( 4759,  2.0),  # Maranhao
            ],
            'delta': [
                ( 4652,  2.0),  # Durango
                ( 4653,  2.0),  # Guanajuato
                ( 4673,  2.0),  # Yucatan
                ( 4752,  1.5),  # Amazonas
                ( 4759,  2.0),  # Maranhao
            ],
            'omicron': [
                (   46,  2.0),  # Croatia
                (   47,  3.0),  # Czechia
                (43858,  2.0),  # Alberta
                (43860,  4.0),  # Manitoba
                (  528,  1.2),  # Colorado
                (  530,  1.2),  # Delaware
                (  532,  1.5),  # Florida
                (  535,  1.2),  # Idaho
                (  536,  1.5),  # Illinois
                (  538,  1.5),  # Iowa
                (  548,  1.5),  # Missouri
                (  550,  1.5),  # Nebraska
                (  551,  1.5),  # Nevada
                (  553,  1.5),  # New Jersey
                (  555,  1.2),  # New York
                (  556,  1.5),  # North Carolina
                (  558,  1.5),  # Ohio
                (  559,  1.5),  # Oklahoma
                (  563,  1.5),  # South Carolina
                (  564,  1.5),  # South Dakota
                (  565,  1.5),  # Tennessee
                (  566,  1.5),  # Texas
                (  567,  1.5),  # Utah
                (   78,  2.0),  # Denmark
                (   84,  1.5),  # Ireland
                (   90,  1.5),  # Norway
                (60368,  2.0),  # Catalonia
                (60373,  2.0),  # Melilla
                (  126,  2.0),  # Costa Rica
                ( 4644,  2.0),  # Baja California
                ( 4647,  3.0),  # Coahuila
                ( 4652,  2.0),  # Durango
                ( 4653,  3.0),  # Guanajuato
                ( 4655,  2.0),  # Hidalgo
                ( 4659,  2.0),  # Morelos
                ( 4661,  2.0),  # Nueva Leone
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  2.0),  # Tabasco
                ( 4670,  2.0),  # Tamaulipas
                ( 4673,  2.5),  # Yucatan
                ( 4753,  0.6),  # Amapa
                ( 4752,  1.5),  # Amazonas
                ( 4756,  1.2),  # Distrito Federal
                ( 4758,  0.8),  # Goias
                ( 4761,  0.8),  # Mato Grosso do Sol
                ( 4759,  2.0),  # Maranhao
                ( 4765,  0.8),  # Parana
                ( 4770,  0.8),  # Rondonia
                (  196,  2.0),  # South Africa
            ],
        }

        if rates_parameters.heavy_hand_fixes:
            for variant, scaling_factors in ihr_scaling_factors.items():
                kappa = pd.Series(
                    sampled_ode_params[f'kappa_{variant}_admission'],
                    index=compartments.reset_index().location_id.unique(),
                    name=f'kappa_{variant}_admission'
                )
                for location_id, scaling_factor in scaling_factors:
                    kappa.loc[location_id] *= scaling_factor
                sampled_ode_params[f'kappa_{variant}_admission'] = kappa

    if measure == 'death':
        ifr_scaling_factors = {
            'ancestral': [
                (   37,  1.5),  # Kyrgyzstan
                (  121,  3.0),  # Bolivia
            ],
            'alpha': [
                (   34,  5.0),  # Azerbaijan
                (   37,  5.0),  # Kyrgyzstan
                (   43,  5.0),  # Albania
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                (  121,  3.0),  # Bolivia
                (  160,  3.0),  # Afghanistan
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  187,  5.0),  # Somalia
                (  173,  5.0),  # Gabon
            ],
            'beta': [
                (  151,  2.0),  # Qatar
                (  177,  3.0),  # Djibouti
                (  184,  3.0),  # Mozambique
                (  193,  2.0),  # Botswana
                (  197,  3.0),  # Eswatini
                (  194,  8.0),  # Lesotho
            ],
            'gamma': [
                (  121,  3.0),  # Bolivia
                ( 4753,  2.0),  # Amapa
                ( 4752,  1.5),  # Amazonas
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                ( 4759,  1.5),  # Maranhao
                ( 4762,  1.5),  # Mato Grosso
            ],
            'delta': [
                (   33,  3.0),  # Armenia
                (   34,  5.0),  # Azerbaijan
                (   35,  3.0),  # Georgia
                (   36, 10.0),  # Kazakhstan
                (   37,  5.0),  # Kyrgyzstan
                (   38,  2.0),  # Mongolia
                (   41,  2.0),  # Uzbekistan
                (   43,  5.0),  # Albania
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                (   57,  5.0),  # Belarus
                (   59,  2.0),  # Latvia
                (   60,  2.0),  # Lithuania
                (   63,  2.0),  # Ukraine
                (  113,  2.0),  # Guyana
                (  118,  2.0),  # Suriname
                (  119,  1.5),  # Trinidad and Tobago
                (  121,  5.0),  # Bolivia
                (  129,  2.0),  # Honduras
                ( 4651,  2.0),  # Mexico city
                ( 4651,  2.0),  # Hidalgo
                ( 4673,  2.0),  # Yucatan
                ( 4753,  2.0),  # Amapa
                ( 4752,  1.5),  # Amazonas
                ( 4755,  1.5),  # Ceara
                ( 4759,  1.5),  # Maranhao
                ( 4762,  1.5),  # Mato Grosso
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                (  136,  1.5),  # Paraguay
                (  160,  3.0),  # Afghanistan
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  147,  2.0),  # Libya
                (  148,  2.0),  # Morocco
                (  151,  2.0),  # Qatar
                ( 4843,  2.0),  # Assam
                ( 4849,  1.5),  # Delhi
                ( 4861,  2.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  2.0),  # Mizoram
                ( 4864,  2.0),  # Nagaland
                ( 4845,  3.0),  # Chandigarh
                ( 4866,  2.0),  # Puducherry
                ( 4874,  2.0),  # Uttarakhand
                (53617,  2.0),  # Gilgit-Baltistan
                (53619,  5.0),  # Khyber Pakhtunkhwa
                (53620,  2.0),  # Punjab
                (53621,  2.0),  # Sindh
                (  168,  3.0),  # Angola
                (  169,  2.0),  # Central African Republic
                (  170,  2.0),  # Congo
                (  173,  3.0),  # Gabon
                (  181,  2.0),  # Madagascar
                (  182,  2.0),  # Malawi
                (  184,  3.0),  # Mozambique
                (  187,  3.0),  # Somalia
                (  190,  2.0),  # Uganda
                (  191,  5.0),  # Zambia
                (  193,  3.0),  # Botswana
                (  197,  3.0),  # Eswatini
                (  194,  5.0),  # Lesotho
                (  195,  3.0),  # Namibia
                (  198, 10.0),  # Zimbabwe
                (  201,  2.0),  # Burkina Faso
                (  205,  2.0),  # Cote d'Ivoire
                (  208,  3.0),  # Guinea
                (  209,  5.0),  # Guinea Bissau
                (  210,  3.0),  # Liberia
                (  212,  2.0),  # Mauritania
                (  216,  2.0),  # Senegal
            ],
            'omicron': [
                (   33,  8.0),  # Armenia
                (   34, 20.0),  # Azerbaijan
                (   35, 25.0),  # Georgia
                (   36, 10.0),  # Kazakhstan
                (   37, 10.0),  # Kyrgyzstan
                (   38,  2.0),  # Mongolia
                (   41,  3.0),  # Uzbekistan
                (   43, 10.0),  # Albania
                (   44, 10.0),  # Bosnia and Herzegovina
                (   45, 10.0),  # Bulgaria
                (   46,  5.0),  # Croatia
                (   47,  4.0),  # Czechia
                (   48,  3.0),  # Hungary
                (   49, 15.0),  # North Macedonia
                (   51,  2.0),  # Poland
                (   52,  3.0),  # Romania
                (   53,  2.0),  # Serbia
                (   54,  2.5),  # Slovakia
                (   57, 10.0),  # Belarus
                (   58,  2.0),  # Estonia
                (   59,  8.0),  # Latvia
                (   60,  4.0),  # Lithuania
                (   61,  1.5),  # Moldova
                (43860,  3.0),  # Manitoba
                (43861,  2.0),  # New Brunswick
                (43862,  2.0),  # Newfoundland and Labrador
                (  523,  1.5),  # Alabama
                (  524,  1.5),  # Alaska
                (  525,  1.5),  # Arizona
                (  526,  1.5),  # Arkansas
                (  528,  1.5),  # Colorado
                (  532,  1.5),  # Florida
                (  533,  1.5),  # Georgia
                (  535,  1.5),  # Idaho
                (  536,  1.5),  # Illinois
                (  537,  1.2),  # Indiana
                (  538,  1.5),  # Iowa
                (  541,  1.2),  # Louisiana
                (  543,  1.5),  # Maryland
                (  545,  1.5),  # Michigan
                (  546,  1.5),  # Minnesota
                (  547,  1.5),  # Mississippi
                (  548,  1.5),  # Missouri
                (  550,  1.5),  # Nebraska
                (  551,  1.5),  # Nevada
                (  553,  1.5),  # New Jersey
                (  554,  1.5),  # New Mexico
                (  555,  1.2),  # New York
                (  556,  1.5),  # North Carolina
                (  558,  1.5),  # Ohio
                (  559,  1.5),  # Oklahoma
                (  561,  1.5),  # Pennsylvania
                (  563,  1.5),  # South Carolina
                (  564,  1.5),  # South Dakota
                (  565,  1.5),  # Tennessee
                (  566,  1.5),  # Texas
                (  567,  1.5),  # Utah
                (  572,  1.5),  # Wisconsin
                (  573,  1.5),  # Wyoming
                (   72,  0.9),  # Andorra
                (   78,  2.0),  # Denmark
                (   85,  2.0),  # Israel
                (35512,  2.0),  # Calabria
                (35513,  2.0),  # Sicilia
                (60364,  2.0),  # Canary Islands
                (60373,  2.0),  # Melilla
                (  122,  2.0),  # Ecuador
                (  123,  1.5),  # Peru
                (  108,  1.5),  # Belize
                (  113,  3.0),  # Guyana
                (  114,  1.5),  # Haiti
                (  115,  1.5),  # Jamaica
                (  115,  1.5),  # Saint Lucia
                (  118,  2.0),  # Suriname
                (  119,  2.0),  # Trinidad and Tobago
                (  422,  2.0),  # US Virgin Islands
                (  121, 10.0),  # Bolivia
                (  129,  3.0),  # Honduras
                ( 4643,  2.0),  # Aguascalientes
                ( 4644,  5.0),  # Baja California
                ( 4645,  2.0),  # Baja California do Sur
                ( 4647,  3.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4656,  2.0),  # Jalisco
                ( 4657,  2.0),  # Mexico state
                ( 4651,  2.0),  # Mexico city
                ( 4655,  3.0),  # Hidalgo
                ( 4659,  2.0),  # Morelos
                ( 4661,  2.0),  # Nueva Leone
                ( 4663,  2.0),  # Puebla
                ( 4665,  2.0),  # Quintana Roo
                ( 4666,  2.0),  # San Luis Potosoi
                ( 4667,  2.0),  # Sinaloa
                ( 4668,  2.0),  # Sonora
                ( 4669,  2.0),  # Tabasco
                ( 4670,  2.0),  # Tamaulipas
                ( 4673,  3.0),  # Yucatan
                ( 4674,  2.0),  # Zacatecas
                ( 4750,  1.5),  # Acre
                ( 4753,  1.5),  # Amapa
                ( 4752,  1.5),  # Amazonas
                ( 4755,  1.5),  # Ceara
                ( 4756,  2.0),  # Distrito Federal
                ( 4757,  2.5),  # Espirito Santo
                ( 4758,  2.0),  # Goias
                ( 4759,  2.0),  # Maranhao
                ( 4762,  1.5),  # Mato Grosso
                ( 4761,  1.5),  # Mato Grosso do Sol
                ( 4760,  1.5),  # Minas Gerais
                ( 4765,  1.2),  # Parana
                ( 4766,  1.2),  # Pernambuco
                ( 4767,  1.2),  # Piaui
                ( 4769,  1.2),  # Rio Grande do Norte
                ( 4768,  1.2),  # Rio de Janeiro
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.5),  # Roraima
                ( 4773,  1.2),  # Santa Catarina
                ( 4773,  2.0),  # Sao Paolo
                ( 4776,  1.5),  # Tocantins
                (  136,  3.0),  # Paraguay
                (  160,  5.0),  # Afghanistan
                (  140,  2.0),  # Bahrain
                (  141,  5.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  144,  3.0),  # Jordan
                (  146,  2.0),  # Lebanon
                (  147,  3.0),  # Libya
                (  148,  3.0),  # Morocco
                (  149,  3.0),  # Palestine
                (  151,  3.0),  # Qatar
                (  522,  2.0),  # Sudan
                (  154,  5.0),  # Tunisia
                (  155,  2.0),  # Turkey
                (  157,  2.0),  # Yemen
                (  162,  2.0),  # Bhutan
                ( 4843,  2.0),  # Assam
                ( 4849,  2.0),  # Delhi
                ( 4851,  1.5),  # Gujarat
                ( 4852,  1.5),  # Haryana
                ( 4853,  1.2),  # Himachal Pradesh
                ( 4854,  1.2),  # Jammu & Kashmir and Ladakh
                ( 4861,  4.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  4.0),  # Mizoram
                ( 4864,  2.0),  # Nagaland
                ( 4865,  3.0),  # Odisha
                ( 4845,  5.0),  # Chandigarh
                ( 4866,  2.0),  # Puducherry
                ( 4867,  2.0),  # Punjab
                ( 4868,  1.5),  # Rajasthan
                ( 4872,  2.0),  # Tripura
                ( 4874,  3.0),  # Uttarakhand
                ( 4875,  1.5),  # West Bengal
                (  164,  2.0),  # Nepal
                (53615,  2.0),  # Azad Jammu & Kashmir
                (53617,  3.0),  # Gilgit-Baltistan
                (53618,  2.0),  # Islamabad Capital Territory
                (53619,  8.0),  # Khyber Pakhtunkhwa
                (53620,  3.0),  # Punjab
                (53621,  2.0),  # Sindh
                (   22,  1.5),  # Fiji
                (  351,  3.0),  # Guam
                (   11,  1.5),  # Indonesia
                (   13,  1.5),  # Malaysia
                (  168,  6.0),  # Angola
                (  169,  3.0),  # Central African Republic
                (  170,  3.0),  # Congo
                (  171,  5.0),  # DRC
                (  172,  3.0),  # Equatorial Guinea
                (  173,  5.0),  # Gabon
                (  178,  2.0),  # Eritrea
                (  179,  2.0),  # Ethiopia
                (  180,  5.0),  # Kenya
                (  181,  5.0),  # Madagascar
                (  182,  3.0),  # Malawi
                (  184,  3.0),  # Mozambique
                (  185,  2.0),  # Rwanda
                (  187,  3.0),  # Somalia
                (  190,  3.0),  # Uganda
                (  191,  8.0),  # Zambia
                (  193,  5.0),  # Botswana
                (  197,  3.0),  # Eswatini
                (  194,  5.0),  # Lesotho
                (  195, 10.0),  # Namibia
                (  196,  3.0),  # South Africa
                (  198, 20.0),  # Zimbabwe
                (  201,  8.0),  # Burkina Faso
                (  203,  1.5),  # Cabo Verde
                (  202,  1.5),  # Cameroon
                (  204,  0.8),  # Chad
                (  205,  4.0),  # Cote d'Ivoire
                (  207,  3.0),  # Ghana
                (  208,  5.0),  # Guinea
                (  209, 20.0),  # Guinea Bissau
                (  210,  5.0),  # Liberia
                (  211,  2.0),  # Mali
                (  212,  3.0),  # Mauritania
                (  213,  2.0),  # Niger
                (  214,  0.2),  # Nigeria
                (  216,  3.0),  # Senegal
            ],
        }

        if rates_parameters.heavy_hand_fixes:
            for variant, scaling_factors in ifr_scaling_factors.items():
                kappa = pd.Series(
                    sampled_ode_params[f'kappa_{variant}_death'],
                    index=compartments.reset_index().location_id.unique(),
                    name=f'kappa_{variant}_death'
                )
                for location_id, scaling_factor in scaling_factors:
                    kappa.loc[location_id] *= scaling_factor
                sampled_ode_params[f'kappa_{variant}_death'] = kappa

    return sampled_ode_params

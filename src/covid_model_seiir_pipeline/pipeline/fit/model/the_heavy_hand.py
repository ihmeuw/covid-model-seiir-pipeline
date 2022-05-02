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
                (   37,  8.0),  # Kyrgyzstan
                (   50,  2.0),  # Montenegro
                (   61,  2.0),  # Moldova
                (  114, 0.25),  # Haiti
                ( 4647,  2.0),  # Coahuila
                ( 4759,  0.5),  # Maranhao
                (  153,  0.5),  # Syrian Arab Republic
                (  168,  1.5),  # Angola
                (  200,  0.5),  # Benin
            ],
            'alpha': [
                (   33,  3.0),  # Armenia
                (   37,  6.0),  # Kyrgyzstan
                (   61,  4.0),  # Moldova
                (  114, 0.25),  # Haiti
                (  121,  3.0),  # Bolivia
                (  139, 0.25),  # Algeria
                (  522, 0.25),  # Sudan
                (  153,  0.1),  # Syrian Arab Republic
                (  157, 0.25),  # Yemen
                (   15, 0.75),  # Myanmar
                (  168,  3.0),  # Angola
                (  201,  3.0),  # Burkina Faso
                (  204, 0.75),  # Chad
            ],
            'beta': [
                (  151, 0.25),  # Qatar
                (  184, 0.25),  # Mozambique
            ],
            'gamma': [
                (  114, 0.25),  # Haiti
                (  121,  3.0),  # Bolivia
                ( 4757,  3.0),  # Espirito Santo
                ( 4759, 0.25),  # Maranhao
                ( 4762,  2.0),  # Mato Grosso
                ( 4761,  2.0),  # Mato Grosso do Sul
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
            ],
            'delta': [
                (   33,  3.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   37,  6.0),  # Kyrgyzstan
                (   38,  3.0),  # Mongolia
                (   43,  3.0),  # Albania
                (   50,  2.0),  # Montenegro
                (   61,  2.0),  # Moldova
                (   62,  2.0),  # Russia
                (  114,  0.1),  # Haiti
                (  115,  0.5),  # Jamaica
                (  121,  3.0),  # Bolivia
                ( 4644,  2.0),  # Baja California
                ( 4647,  2.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  2.0),  # Tabasco
                ( 4673,  3.0),  # Yucatan
                ( 4757,  3.0),  # Espirito Santo
                ( 4759, 0.25),  # Maranhao
                ( 4762,  2.0),  # Mato Grosso
                ( 4761,  2.0),  # Mato Grosso do Sul
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
                (  139,  0.1),  # Algeria
                (  147,  2.0),  # Libya
                (  151, 0.25),  # Qatar
                (  152, 0.25),  # Saudi Arabia
                (  522, 0.25),  # Sudan
                (  153,  0.1),  # Syrian Arab Republic
                (  157, 0.05),  # Yemen
                (  161,  2.0),  # Bangladesh
                ( 4842,  2.0),  # Arunachal Pradesh
                ( 4849,  2.0),  # Delhi
                ( 4860,  0.5),  # Maharashtra
                ( 4861,  3.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  8.0),  # Mizoram
                ( 4845,  3.0),  # Chandigarh
                ( 4858,  5.0),  # Lakshadweep
                ( 4866,  5.0),  # Puducherry
                ( 4869,  2.0),  # Sikkim
                ( 4870,  0.5),  # Tamil Nadu
                ( 4873,  0.5),  # Uttar Pradesh
                (53619,  2.0),  # Khyber Pakhtunkhwa
                (   15, 0.75),  # Myanmar
                (  186,  3.0),  # Seychelles
                (   19, 0.75),  # Timor-Leste
                (  168,  3.0),  # Angola
                (  169,  1.5),  # Central African Republic
                (  171,  2.0),  # DRC
                (  172,  0.5),  # Equatorial Guinea
                (  173,  0.5),  # Gabon
                (  181,  2.0),  # Madagascar
                (  184, 0.25),  # Mozambique
                (  185,  0.5),  # Rwanda
                (  198, 1.25),  # Zimbabwe
                (  200,  0.5),  # Benin
                (  204, 0.75),  # Chad
            ],
            'omicron': [
                (   33,  1.5),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   35,  2.5),  # Georgia
                (   36,  0.5),  # Kazakhstan
                (   37,  5.0),  # Kyrgyzstan
                (   38,  3.0),  # Mongolia
                (   41, 0.25),  # Uzbekistan
                (   43,  3.0),  # Albania
                (   46,  1.5),  # Croatia
                (   50,  2.0),  # Montenegro
                (   49,  0.8),  # North Macedonia
                (   51,  2.0),  # Poland
                (   53,  2.0),  # Serbia
                (   57,  0.5),  # Belarus
                (   58,  2.0),  # Estonia
                (   59,  2.0),  # Latvia
                (   60,  3.0),  # Lithuania
                (   61,  4.0),  # Moldova
                (   62,  5.0),  # Russia
                (   68,  3.0),  # Republic of Korea
                (   69, 0.75),  # Singapore
                (43860,  2.0),  # Manitoba
                (  531,  2.0),  # District of Columbia
                (  536,  1.5),  # Illinois
                (  537, 0.75),  # Indiana
                (  541, 0.75),  # Louisiana
                (  542,  0.5),  # Maine
                (  560,  0.5),  # Oregon
                (  568,  0.5),  # Vermont
                (60886,  0.5),  # King and Snohomish Counties
                (   97,  0.5),  # Argentina
                (   74,  1.5),  # Andorra
                (   76,  1.2),  # Belgium
                (   78,  2.0),  # Denmark
                (60377,  2.0),  # Baden Wurttemberg
                (60378,  3.0),  # Bavaria
                (60379,  2.0),  # Berlin
                (60380,  2.0),  # Brandenberg
                (60381,  2.0),  # Bremen
                (60382,  2.0),  # Hamburg
                (60383,  2.0),  # Hesse
                (60384,  2.0),  # Lower Saxony
                (60385,  2.0),  # Mecklenburg-Vorpommern
                (60386,  2.0),  # North Rhine-Westphalia
                (60387,  2.0),  # Rhineland-Palatinate
                (60388,  2.0),  # Saarland
                (60390,  2.0),  # Saxony
                (60389,  2.0),  # Saxony-Anhalt
                (60391,  2.0),  # Schleswig-Holstein
                (60392,  2.0),  # Thuringia
                (   83,  2.0),  # Iceland
                (   90,  2.0),  # Norway
                (   94,  2.0),  # Switzerland
                (  396,  3.0),  # San Marino
                (60357,  0.5),  # Andalucia
                (60358,  1.5),  # Aragon
                (60365,  1.5),  # Asturias
                (60363,  2.0),  # Balearic Islands
                (60367,  1.5),  # Canary Islands
                (60359,  0.5),  # Cantabria
                (60366, 0.75),  # Murcia
                (60370,  2.0),  # Navarre
                (  106,  0.5),  # Bahamas
                (  107,  0.5),  # Barbados
                (  108,  0.5),  # Belize
                (  305,  0.5),  # Bermuda
                (  109,  0.1),  # Cuba
                (  110, 0.25),  # Dominica
                (  111,  0.5),  # Dominican Republic
                (  113, 0.25),  # Guyana
                (  114, 0.01),  # Haiti
                (  115,  0.1),  # Jamaica
                (  385,  0.5),  # Puerto Rico
                (  121,  4.0),  # Bolivia
                (  122,  1.5),  # Ecuador
                (  125,  0.5),  # Colombia
                (  127, 0.25),  # El Salvador
                (  128,  0.5),  # Guatemala
                (  129, 0.25),  # Honduras
                ( 4644,  2.0),  # Baja California
                ( 4650,  0.5),  # Chihuahua
                ( 4647,  2.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4654,  0.5),  # Guerrero
                ( 4656, 0.75),  # Jalisco
                ( 4658,  0.5),  # Michoacan de Ocampo
                ( 4659,  0.5),  # Morelos
                ( 4664, 0.75),  # Queretaro
                ( 4665,  2.0),  # Quintana Roo
                ( 4667,  2.0),  # Sinaloa
                ( 4669,  3.0),  # Tabasco
                ( 4673,  4.0),  # Yucatan
                ( 4674, 0.75),  # Zacatecas
                ( 4752,  0.5),  # Amazonas
                ( 4755,  0.5),  # Ceara
                ( 4756,  1.2),  # Distrito Federal
                ( 4757,  6.0),  # Espirito Santo
                ( 4758,  0.8),  # Goias
                ( 4759, 0.25),  # Maranhao
                ( 4762,  2.0),  # Mato Grosso
                ( 4761,  2.0),  # Mato Grosso do Sul
                ( 4760,  2.0),  # Minas Gerais
                ( 4764,  2.0),  # Paraiba
                ( 4765,  0.8),  # Parana
                ( 4767, 0.25),  # Piaui
                ( 4769,  0.8),  # Rio Grande do Norte
                ( 4772,  0.5),  # Rio Grande do Sul
                ( 4771,  2.0),  # Roraima
                ( 4773,  0.5),  # Sao Paolo
                ( 4774,  0.1),  # Sergipe
                (  136,  0.5),  # Paraguay
                (  160, 0.25),  # Afghanistan
                (  139,0.005),  # Algeria
                (  140,  1.5),  # Bahrain
                (  141,  0.5),  # Egypt
                (  142,  0.3),  # Iran
                (  143,  0.5),  # Iraq
                (  144,  2.0),  # Jordan
                (  145,  0.5),  # Kuwait
                (  147,  1.5),  # Libya
                (  148,  0.5),  # Morocco
                (  151, 0.25),  # Qatar
                (  152, 0.02),  # Saudi Arabia
                (  522, 0.02),  # Sudan
                (  153,0.005),  # Syrian Arab Republic
                (  155,  2.0),  # Turkey
                (  156,  0.1),  # United Arab Emirates
                (  157,0.001),  # Yemen
                (  161,  1.5),  # Bangladesh
                ( 4841,  0.5),  # Andhra Pradesh
                ( 4842,  1.5),  # Arunachal Pradesh
                ( 4843,  0.5),  # Assam
                ( 4844,  0.5),  # Bihar
                ( 4846,  0.5),  # Chhattisgarh
                ( 4849,  3.0),  # Delhi
                ( 4850, 0.25),  # Goa
                ( 4851,  1.5),  # Gujarat
                ( 4855,  0.5),  # Jharkhand
                ( 4856, 0.25),  # Karnataka
                ( 4857,  0.5),  # Kerala
                ( 4860,  0.1),  # Maharashtra
                ( 4861,  2.0),  # Manipur
                ( 4862,  1.5),  # Meghalaya
                ( 4863, 50.0),  # Mizoram
                ( 4864,  0.5),  # Nagaland
                ( 4845,  5.0),  # Chandigarh
                ( 4858,  5.0),  # Lakshadweep
                ( 4866, 10.0),  # Puducherry
                ( 4867, 0.75),  # Punjab
                ( 4868,  0.5),  # Rajasthan
                ( 4869,  3.0),  # Sikkim
                ( 4870,  0.1),  # Tamil Nadu
                ( 4871,  0.5),  # Telengana
                ( 4873,  0.2),  # Uttar Pradesh
                ( 4875,  0.5),  # West Bengal
                (  164,  0.5),  # Nepal
                (53617,  0.8),  # Gilgit-Baltistan
                (53619,  2.0),  # Khyber Pakhtunkhwa
                (53620,  0.5),  # Punjab
                (  351,  5.0),  # Guam
                (   26, 0.25),  # Papua New Guinea
                (   10,  0.1),  # Cambodia
                (   14,  1.5),  # Maldives
                (   15,  0.1),  # Myanmar
                (   17,  0.1),  # Sri Lanka
                (  186,  3.0),  # Seychelles
                (   19,  0.1),  # Timor-Leste
                (   20,  6.0),  # Viet Nam
                (  168,  3.0),  # Angola
                (  169,  1.5),  # Central African Republic
                (  170, 0.25),  # Congo
                (  171,  4.0),  # DRC
                (  172, 0.25),  # Equatorial Guinea
                (  173, 0.25),  # Gabon
                (  177, 0.25),  # Djibouti
                (  178, 0.25),  # Eritrea
                (  184, 0.05),  # Mozambique
                (  185, 0.25),  # Rwanda
                (  187, 0.25),  # Somalia
                (  190, 0.25),  # Uganda
                (  194, 0.25),  # Lesotho
                (  195,  0.5),  # Namibia
                (  197, 0.75),  # Eswatini
                (  198,  1.5),  # Zimbabwe
                (  200, 0.01),  # Benin
                (  201, 0.75),  # Burkina Faso
                (  203,  1.5),  # Cabo Verde
                (  202, 0.25),  # Cameroon
                (  204,  0.1),  # Chad
                (  205,  0.5),  # Cote d'Ivoire
                (  206,  0.5),  # Gambia
                (  207,  0.5),  # Ghana
                (  208,  0.5),  # Guinea
                (  209, 0.25),  # Guinea-Bissau
                (  210,  0.5),  # Liberia
                (  211,  2.0),  # Mali
                (  212,  0.5),  # Mauritania
                (  213,  0.2),  # Niger
                (  214,  0.5),  # Nigeria
                (  216, 0.25),  # Senegal
                (  217,  0.1),  # Sierra Leone
                (  218,  0.5),  # Togo
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
            'ancestral': [
                ( 4647,  2.0),  # Coahuila
                ( 4754,  0.5),  # Bahia
            ],
            'alpha': [
                (   51,  0.5),  # Poland
                (  544,  2.0),  # Massachusetts
            ],
            'gamma': [
                ( 4752,  1.5),  # Amazonas
                ( 4754,  0.5),  # Bahia
                ( 4759,  2.0),  # Maranhao
            ],
            'delta': [
                (   51,  0.5),  # Poland
                (  544,  2.0),  # Massachusetts
                ( 4647,  2.0),  # Coahuila
                ( 4652,  2.0),  # Durango
                ( 4653,  2.0),  # Guanajuato
                ( 4673,  2.0),  # Yucatan
                ( 4752,  1.5),  # Amazonas
                ( 4754,  0.5),  # Bahia
                ( 4759,  2.0),  # Maranhao
            ],
            'omicron': [
                (   46,  1.5),  # Croatia
                (   47,  3.0),  # Czechia
                (   51,  0.5),  # Poland
                (43858,  2.0),  # Alberta
                (43860,  4.0),  # Manitoba
                (  528,  1.2),  # Colorado
                (  530,  1.2),  # Delaware
                (  532,  1.5),  # Florida
                (  534,  0.5),  # Hawaii
                (  535,  1.2),  # Idaho
                (  536,  1.5),  # Illinois
                (  538,  1.5),  # Iowa
                (  544,  3.0),  # Massachusetts
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
                (  567,  2.0),  # Utah
                (  568,  0.5),  # Vermont
                (60886,  0.5),  # King and Snohomish Counties
                (   97, 0.25),  # Argentina
                (   98,  0.5),  # Chile
                (   78,  2.0),  # Denmark
                (   84,  1.5),  # Ireland
                (60357,  0.5),  # Andalucia
                (60367,  0.5),  # Castile and Leon
                (60368,  2.0),  # Catalonia
                (60362,  0.5),  # Extremadura
                (60372,  0.5),  # Galicia
                (60376,  0.5),  # La Rioja
                (60373,  2.0),  # Melilla
                (60370,  0.5),  # Navarre
                (60371, 0.75),  # Valencian Community
                (   93, 0.75),  # Sweden
                (   94, 0.75),  # Switzerland
                (  385,  0.5),  # Puerto Rico
                (  126,  3.0),  # Costa Rica
                ( 4643, 0.75),  # Aguascalientes
                ( 4644,  3.0),  # Baja California
                ( 4645,  1.5),  # Baja California Sur
                ( 4646,  0.5),  # Campeche
                ( 4755,  0.5),  # Ceara
                ( 4649, 0.25),  # Chiapas
                ( 4650,  0.5),  # Chihuahua
                ( 4647,  1.5),  # Coahuila
                ( 4648,  0.5),  # Colima
                ( 4652,  2.0),  # Durango
                ( 4653,  3.0),  # Guanajuato
                ( 4654,  0.5),  # Guerrero
                ( 4655,  2.0),  # Hidalgo
                ( 4657, 0.75),  # Mexico
                ( 4658, 0.75),  # Michoacan de Ocampo
                ( 4659,  2.0),  # Morelos
                ( 4661,  2.0),  # Nueva Leone
                ( 4660, 0.75),  # Nayarit
                ( 4662, 0.75),  # Oaxaca
                ( 4664,  0.5),  # Queretaro
                ( 4665,  2.0),  # Quintana Roo
                ( 4666, 0.75),  # San Luis Potosi
                ( 4669,  2.0),  # Tabasco
                ( 4670,  2.0),  # Tamaulipas
                ( 4673,  2.5),  # Yucatan
                ( 4674, 0.75),  # Zacatecas
                ( 4751,  0.6),  # Alagoas
                ( 4753,  0.6),  # Amapa
                ( 4754,  0.5),  # Bahia
                ( 4756,  1.2),  # Distrito Federal
                ( 4757,  0.6),  # Espirito Santo
                ( 4758,  0.8),  # Goias
                ( 4761,  0.8),  # Mato Grosso do Sol
                ( 4759,  2.0),  # Maranhao
                ( 4763,  0.2),  # Para
                ( 4764,  0.5),  # Paraiba
                ( 4765,  0.8),  # Parana
                ( 4766,  0.1),  # Pernambuco
                ( 4767,  0.2),  # Piaui
                ( 4769,  0.5),  # Rio Grande do Norte
                ( 4772,  0.5),  # Rio Grande do Sul
                ( 4768,  0.8),  # Rio de Janeiro
                ( 4771,  0.5),  # Roraima
                ( 4774,  0.5),  # Sergipe
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
                (   34,  2.0),  # Azerbaijan
                (   37,  3.0),  # Kyrgyzstan
                (  121,  3.0),  # Bolivia
                ( 4647,  2.0),  # Coahuila
                (  168,  2.0),  # Angola
                (  169,  2.0),  # Central African Republic
                (  206,  1.5),  # Gambia
            ],
            'alpha': [
                (   34,  3.0),  # Azerbaijan
                (   37,  2.0),  # Kyrgyzstan
                (   43,  5.0),  # Albania
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                (  562,  0.5),  # Rhode Island
                (  121,  3.0),  # Bolivia
                (  160,  3.0),  # Afghanistan
                (  139,  0.5),  # Algeria
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (   15, 0.75),  # Myanmar
                (  168,  2.0),  # Angola
                (  187,  5.0),  # Somalia
                (  173,  5.0),  # Gabon
                (  204, 0.75),  # Chad
            ],
            'beta': [
                (  151,  2.0),  # Qatar
                (  177,  3.0),  # Djibouti
                (  181,  2.0),  # Madagascar
                (  184,  2.0),  # Mozambique
                (  193,  2.0),  # Botswana
                (  197,  5.0),  # Eswatini
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
                (   34,  3.0),  # Azerbaijan
                (   35,  3.0),  # Georgia
                (   36, 10.0),  # Kazakhstan
                (   37,  2.0),  # Kyrgyzstan
                (   38,  2.0),  # Mongolia
                (   41,  2.0),  # Uzbekistan
                (   43,  5.0),  # Albania
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                (   57,  5.0),  # Belarus
                (   59,  2.0),  # Latvia
                (   60,  2.0),  # Lithuania
                (   63,  2.0),  # Ukraine
                (  562,  0.5),  # Rhode Island
                ( 4749,  0.5),  # England
                (  433,  0.5),  # Northern Ireland
                (  434,  0.5),  # Scotland
                ( 4636,  0.5),  # Wales
                (  113,  2.0),  # Guyana
                (  118,  2.0),  # Suriname
                (  119,  1.5),  # Trinidad and Tobago
                (  121,  5.0),  # Bolivia
                (  129,  2.0),  # Honduras
                ( 4647,  2.0),  # Coahuila
                ( 4655,  2.0),  # Hidalgo
                ( 4651,  2.0),  # Mexico city
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
                (  139,  0.5),  # Algeria
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  147,  2.0),  # Libya
                (  148,  2.0),  # Morocco
                (  151,  2.0),  # Qatar
                (  153,  0.5),  # Syrian Arab Republic
                ( 4843,  2.0),  # Assam
                ( 4849,  1.5),  # Delhi
                ( 4859,  0.5),  # Madhya Pradesh
                ( 4860,  0.5),  # Maharashtra
                ( 4861,  2.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  2.0),  # Mizoram
                ( 4864,  2.0),  # Nagaland
                ( 4845,  3.0),  # Chandigarh
                ( 4866,  2.0),  # Puducherry
                ( 4873,  0.5),  # Uttar Pradesh
                ( 4874,  2.0),  # Uttarakhand
                (53617,  2.0),  # Gilgit-Baltistan
                (53619,  5.0),  # Khyber Pakhtunkhwa
                (53620,  2.0),  # Punjab
                (53621,  2.0),  # Sindh
                (   15, 0.75),  # Myanmar
                (   19, 0.75),  # Timor-Leste
                (  168,  2.0),  # Angola
                (  169,  2.0),  # Central African Republic
                (  170,  2.0),  # Congo
                (  173,  3.0),  # Gabon
                (  181,  3.0),  # Madagascar
                (  184,  2.0),  # Mozambique
                (  187,  3.0),  # Somalia
                (  191,  5.0),  # Zambia
                (  193,  3.0),  # Botswana
                (  197,  5.0),  # Eswatini
                (  194,  5.0),  # Lesotho
                (  195,  3.0),  # Namibia
                (  198, 15.0),  # Zimbabwe
                (  200,  0.5),  # Benin
                (  201,  2.0),  # Burkina Faso
                (  204, 0.75),  # Chad
                (  205,  2.0),  # Cote d'Ivoire
                (  206, 10.0),  # Gambia
                (  208,  3.0),  # Guinea
                (  209,  3.0),  # Guinea-Bissau
                (  210,  2.0),  # Liberia
                (  212,  2.0),  # Mauritania
                (  214, 0.75),  # Nigeria
                (  216,  2.0),  # Senegal
            ],
            'omicron': [
                (   33,  8.0),  # Armenia
                (   34, 10.0),  # Azerbaijan
                (   35, 20.0),  # Georgia
                (   36, 10.0),  # Kazakhstan
                (   37,  2.0),  # Kyrgyzstan
                (   38,  2.0),  # Mongolia
                (   41,  3.0),  # Uzbekistan
                (   43, 10.0),  # Albania
                (   44, 10.0),  # Bosnia and Herzegovina
                (   45, 10.0),  # Bulgaria
                (   46,  3.0),  # Croatia
                (   47,  4.0),  # Czechia
                (   48,  3.0),  # Hungary
                (   49, 15.0),  # North Macedonia
                (   51,  3.0),  # Poland
                (   52,  3.0),  # Romania
                (   53,  3.0),  # Serbia
                (   54,  2.5),  # Slovakia
                (   57, 10.0),  # Belarus
                (   58,  2.0),  # Estonia
                (   59,  8.0),  # Latvia
                (   60,  4.0),  # Lithuania
                (   61,  2.0),  # Moldova
                (   68,  3.0),  # Republic of Korea
                (   69, 0.75),  # Singapore
                (43860,  3.0),  # Manitoba
                (43861,  2.0),  # New Brunswick
                (43862,  2.0),  # Newfoundland and Labrador
                (  523,  1.5),  # Alabama
                (  524,  1.5),  # Alaska
                (  525,  1.5),  # Arizona
                (  526,  1.5),  # Arkansas
                (  528,  1.5),  # Colorado
                (  532,  2.0),  # Florida
                (  533,  2.0),  # Georgia
                (  535,  2.0),  # Idaho
                (  536,  2.0),  # Illinois
                (  537,  1.2),  # Indiana
                (  538,  2.0),  # Iowa
                (  539,  1.5),  # Kansas
                (  540,  1.5),  # Kentucky
                (  541,  1.2),  # Louisiana
                (  543,  1.5),  # Maryland
                (  545,  1.5),  # Michigan
                (  546,  1.5),  # Minnesota
                (  547,  1.5),  # Mississippi
                (  548,  2.0),  # Missouri
                (  550,  2.0),  # Nebraska
                (  551,  2.0),  # Nevada
                (  553,  1.5),  # New Jersey
                (  554,  1.5),  # New Mexico
                (  555,  1.2),  # New York
                (  556,  1.5),  # North Carolina
                (  557, 0.75),  # North Dakota
                (  558,  2.0),  # Ohio
                (  559,  2.0),  # Oklahoma
                (  561,  1.5),  # Pennsylvania
                (  562,  0.5),  # Rhode Island
                (  563,  1.5),  # South Carolina
                (  564,  1.5),  # South Dakota
                (  565,  2.0),  # Tennessee
                (  566,  2.0),  # Texas
                (  567,  2.0),  # Utah
                (  568,  0.5),  # Vermont
                (60886,  0.5),  # King and Snohomish Counties
                (  571,  1.5),  # West Virginia
                (  572,  2.0),  # Wisconsin
                (  573,  2.0),  # Wyoming
                (   97,  0.5),  # Argentina
                (   78,  2.0),  # Denmark
                (   79,  2.0),  # Finland
                (   85,  2.0),  # Israel
                (35512,  2.0),  # Calabria
                (35513,  2.0),  # Sicilia
                (  396, 0.25),  # San Marino
                (60357,  0.5),  # Andalucia
                (60365,  0.5),  # Asturias
                (60364,  1.5),  # Canary Islands
                (60376,  0.5),  # La Rioja
                (60373,  2.0),  # Melilla
                (60366, 0.75),  # Murcia
                (60371, 0.75),  # Valencian Community
                (   94, 0.75),  # Switzerland
                ( 4749,  0.5),  # England
                (  433,  0.5),  # Northern Ireland
                (  434,  0.5),  # Scotland
                ( 4636,  0.5),  # Wales
                (  109, 0.25),  # Cuba
                (  111,  0.5),  # Dominican Republic
                (  122,  2.0),  # Ecuador
                (  108,  1.5),  # Belize
                (  112,  0.5),  # Grenada
                (  113,  3.0),  # Guyana
                (  114,  1.5),  # Haiti
                (  385,  0.5),  # Puerto Rico
                (  118,  2.0),  # Suriname
                (  119,  2.0),  # Trinidad and Tobago
                (  422,  2.0),  # US Virgin Islands
                (  121,  8.0),  # Bolivia
                (  129,  3.0),  # Honduras
                ( 4643,  1.5),  # Aguascalientes
                ( 4644,  6.0),  # Baja California
                ( 4645,  3.0),  # Baja California do Sur
                ( 4649,  0.5),  # Chiapas
                ( 4647,  4.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4656,  2.0),  # Jalisco
                ( 4657,  2.0),  # Mexico state
                ( 4651,  2.0),  # Mexico city
                ( 4655,  4.0),  # Hidalgo
                ( 4659,  3.0),  # Morelos
                ( 4661,  2.0),  # Nueva Leone
                ( 4663,  2.0),  # Puebla
                ( 4665,  2.0),  # Quintana Roo
                ( 4666,  2.0),  # San Luis Potosi
                ( 4667,  2.0),  # Sinaloa
                ( 4668,  2.0),  # Sonora
                ( 4669,  2.0),  # Tabasco
                ( 4670,  2.0),  # Tamaulipas
                ( 4673,  3.0),  # Yucatan
                ( 4750,  1.5),  # Acre
                ( 4753,  1.5),  # Amapa
                ( 4752,  1.5),  # Amazonas
                ( 4755,  2.0),  # Ceara
                ( 4756,  2.0),  # Distrito Federal
                ( 4757,  3.0),  # Espirito Santo
                ( 4758,  3.0),  # Goias
                ( 4759,  2.0),  # Maranhao
                ( 4762,  1.5),  # Mato Grosso
                ( 4761,  1.5),  # Mato Grosso do Sol
                ( 4760,  1.5),  # Minas Gerais
                ( 4765,  1.2),  # Parana
                ( 4766,  1.2),  # Pernambuco
                ( 4767,  1.2),  # Piaui
                ( 4769,  0.8),  # Rio Grande do Norte
                ( 4768,  1.5),  # Rio de Janeiro
                ( 4770,  3.0),  # Rondonia
                ( 4771,  2.5),  # Roraima
                ( 4773,  2.0),  # Sao Paolo
                ( 4774,  0.5),  # Sergipe
                ( 4776,  1.5),  # Tocantins
                (  136,  4.0),  # Paraguay
                (  160,  5.0),  # Afghanistan
                (  139,  0.1),  # Algeria
                (  140,  1.5),  # Bahrain
                (  141,  4.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  144,  3.0),  # Jordan
                (  145,  0.5),  # Kuwait
                (  146,  3.0),  # Lebanon
                (  147,  3.0),  # Libya
                (  148,  2.0),  # Morocco
                (  149,  3.0),  # Palestine
                (  151,  3.0),  # Qatar
                (  152,  0.2),  # Saudi Arabia
                (  522,  2.0),  # Sudan
                (  153, 0.25),  # Syrian Arab Republic
                (  154,  5.0),  # Tunisia
                (  155,  2.0),  # Turkey
                (  156,  0.5),  # United Arab Emirates
                (  157,  2.0),  # Yemen
                (  162,  2.0),  # Bhutan
                ( 4841,  0.5),  # Andhra Pradesh
                ( 4843,  1.5),  # Assam
                ( 4844,  0.5),  # Bihar
                ( 4849,  2.0),  # Delhi
                ( 4850,  0.5),  # Goa
                ( 4851,  1.5),  # Gujarat
                ( 4852,  1.5),  # Haryana
                ( 4853,  1.2),  # Himachal Pradesh
                ( 4855,  0.5),  # Jharkhand
                ( 4859,  0.5),  # Madhya Pradesh
                ( 4860, 0.25),  # Maharashtra
                ( 4861,  3.0),  # Manipur
                ( 4862,  1.5),  # Meghalaya
                ( 4863,  4.0),  # Mizoram
                ( 4864,  2.0),  # Nagaland
                ( 4865,  3.0),  # Odisha
                ( 4845,  5.0),  # Chandigarh
                ( 4854,  1.2),  # Jammu & Kashmir and Ladakh
                ( 4866,  2.0),  # Puducherry
                ( 4867,  1.2),  # Punjab
                ( 4868,  1.5),  # Rajasthan
                ( 4870,  0.5),  # Tamil Nadu
                ( 4871,  0.5),  # Telengana
                ( 4872,  2.0),  # Tripura
                ( 4873,  0.5),  # Uttar Pradesh
                ( 4874,  2.0),  # Uttarakhand
                ( 4875,  1.5),  # West Bengal
                (  164,  1.5),  # Nepal
                (53615,  2.0),  # Azad Jammu & Kashmir
                (53617,  3.0),  # Gilgit-Baltistan
                (53618,  2.0),  # Islamabad Capital Territory
                (53619,  8.0),  # Khyber Pakhtunkhwa
                (53620,  1.5),  # Punjab
                (53621,  2.0),  # Sindh
                (  354,  2.0),  # Hong Kong Special Administrative Region of China
                (   22,  1.5),  # Fiji
                (  351,  3.0),  # Guam
                (   10,  0.1),  # Cambodia
                (   11,  1.5),  # Indonesia
                (   13,  1.5),  # Malaysia
                (   15,  0.1),  # Myanmar
                (   17,  0.1),  # Sri Lanka
                (   19, 0.25),  # Timor-Leste
                (  168,  3.0),  # Angola
                (  169,  4.0),  # Central African Republic
                (  170,  2.0),  # Congo
                (  171,  3.0),  # DRC
                (  172,  3.0),  # Equatorial Guinea
                (  173,  6.0),  # Gabon
                (  178,  2.0),  # Eritrea
                (  179,  2.0),  # Ethiopia
                (  180,  3.0),  # Kenya
                (  181, 10.0),  # Madagascar
                (  182,  5.0),  # Malawi
                (  187,  2.0),  # Somalia
                (  191,  6.0),  # Zambia
                (  193,  5.0),  # Botswana
                (  195,  8.0),  # Namibia
                (  196,  3.0),  # South Africa
                (  198, 20.0),  # Zimbabwe
                (  200, 0.25),  # Benin
                (  201,  8.0),  # Burkina Faso
                (  202,  1.5),  # Cameroon
                (  204,  0.5),  # Chad
                (  205,  3.0),  # Cote d'Ivoire
                (  206,  8.0),  # Gambia
                (  207,  2.0),  # Ghana
                (  208,  4.0),  # Guinea
                (  209, 10.0),  # Guinea-Bissau
                (  211,  2.0),  # Mali
                (  212,  3.0),  # Mauritania
                (  213,  2.0),  # Niger
                (  214,  0.5),  # Nigeria
                (  216,  2.0),  # Senegal
                (  217,  0.5),  # Sierra Leone
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

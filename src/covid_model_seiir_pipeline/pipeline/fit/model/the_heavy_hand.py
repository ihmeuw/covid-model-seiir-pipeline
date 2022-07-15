from typing import Dict

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import sample_idr_parameters
from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters


def rescale_kappas(
    measure: str,
    sampled_ode_params: Dict,
    compartments: pd.DataFrame,
    rates_parameters: RatesParameters,
    draw_id: int,
) -> Dict:

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
                ( 4658, 0.75),  # Michoacan de Ocampo
                ( 4659, 0.75),  # Morelos
                ( 4751, 10.0),  # Alagoas
                ( 4753, 10.0),  # Amapa
                ( 4756,  5.0),  # Distrito Federal
                ( 4752,  0.5),  # Amazonas
                ( 4759,  0.5),  # Maranhao
                ( 4762, 10.0),  # Mato Grosso
                ( 4761, 10.0),  # Mato Grosso do Sul
                ( 4763, 10.0),  # Para
                ( 4768,  2.5),  # Rio de Janeiro
                (  153,  0.5),  # Syrian Arab Republic
                (  168, 1.25),  # Angola
                (  200,  0.5),  # Benin
                (  210,  2.5),  # Liberia
            ],
            'alpha': [
                (   33,  3.0),  # Armenia
                (   37,  6.0),  # Kyrgyzstan
                (   55, 0.75),  # Slovenia
                (   61,  4.0),  # Moldova
                (43866, 0.75),  # Ontario
                (43868, 0.75),  # Quebec
                (   78, 0.75),  # Denmark
                (  114, 0.25),  # Haiti
                (  121,  3.0),  # Bolivia
                (  139, 0.25),  # Algeria
                (  146, 0.75),  # Lebanon
                (  150, 0.75),  # Oman
                (  522,  0.1),  # Sudan
                (  153,  0.1),  # Syrian Arab Republic
                (  156, 0.75),  # United Arab Emirates
                (  157, 0.25),  # Yemen
                (   15, 0.75),  # Myanmar
                (  168, 1.25),  # Angola
                (  201,  3.0),  # Burkina Faso
                (  204, 0.75),  # Chad
                (  210,  2.5),  # Liberia
            ],
            'beta': [
                (  151,  0.1),  # Qatar
                (  184, 0.25),  # Mozambique
                (  193,  0.5),  # Botswana
                (  194, 0.75),  # Lesotho
            ],
            'gamma': [
                (  114, 0.25),  # Haiti
                (  121,  3.0),  # Bolivia
                ( 4665,  2.0),  # Quintana Roo
                ( 4753,  1.5),  # Amapa
                ( 4752,  0.1),  # Amazonas
                ( 4755,  0.6),  # Ceara
                ( 4757,  2.5),  # Espirito Santo
                ( 4759, 0.25),  # Maranhao
                ( 4762,  2.0),  # Mato Grosso
                ( 4761,  2.0),  # Mato Grosso do Sul
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
                (  136,  0.5),  # Paraguay
            ],
            'delta': [
                (   33,  2.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   37,  6.0),  # Kyrgyzstan
                (   38,  3.0),  # Mongolia
                (   43,  3.0),  # Albania
                (   50,  2.0),  # Montenegro
                (   55, 0.75),  # Slovenia
                (   61,  2.0),  # Moldova
                (   62,  2.0),  # Russia
                (43866, 0.75),  # Ontario
                (43868, 0.75),  # Quebec
                (   78, 0.75),  # Denmark
                (  106,  0.5),  # Bahamas
                (  112, 0.75),  # Grenada
                (  114,  0.1),  # Haiti
                (  115,  0.5),  # Jamaica
                (  121,  3.0),  # Bolivia
                ( 4644,  2.0),  # Baja California
                ( 4647,  2.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4658,  0.5),  # Michoacan de Ocampo
                ( 4659,  0.5),  # Morelos
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  2.5),  # Tabasco
                ( 4673,  3.0),  # Yucatan
                ( 4753,  1.5),  # Amapa
                ( 4752,  0.1),  # Amazonas
                ( 4755,  0.6),  # Ceara
                ( 4757,  2.5),  # Espirito Santo
                ( 4759, 0.25),  # Maranhao
                ( 4762,  2.0),  # Mato Grosso
                ( 4761,  2.0),  # Mato Grosso do Sul
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
                (  136,  0.5),  # Paraguay
                (  139, 0.25),  # Algeria
                (  146, 0.75),  # Lebanon
                (  147,  2.0),  # Libya
                (  150, 0.75),  # Oman
                (  151,  0.1),  # Qatar
                (  152, 0.25),  # Saudi Arabia
                (  522,  0.1),  # Sudan
                (  153,  0.1),  # Syrian Arab Republic
                (  156, 0.75),  # United Arab Emirates
                (  157, 0.05),  # Yemen
                (  161,  2.0),  # Bangladesh
                ( 4842,  2.0),  # Arunachal Pradesh
                ( 4844,  0.8),  # Bihar
                ( 4846,  0.8),  # Chhattisgarh
                ( 4849, 1.75),  # Delhi
                ( 4852,  0.8),  # Haryana
                ( 4855,  0.8),  # Jharkhand
                ( 4856,  0.5),  # Karnataka
                ( 4857,  0.8),  # Kerala
                ( 4860,  0.5),  # Maharashtra
                ( 4861,  2.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  8.0),  # Mizoram
                ( 4864,  0.8),  # Nagaland
                ( 4845,  3.0),  # Chandigarh
                ( 4858,  5.0),  # Lakshadweep
                ( 4866,  5.0),  # Puducherry
                ( 4869,  2.0),  # Sikkim
                ( 4870,  0.6),  # Tamil Nadu
                ( 4873,  0.6),  # Uttar Pradesh
                ( 4875,  0.8),  # West Bengal
                (53618,  0.6),  # Islamabad Capital Territory
                (53619,  1.5),  # Khyber Pakhtunkhwa
                (53620,  0.8),  # Punjab
                (   15, 0.75),  # Myanmar
                (  186,  3.0),  # Seychelles
                (   19, 0.75),  # Timor-Leste
                (  168, 1.25),  # Angola
                (  169,  1.5),  # Central African Republic
                (  171,  2.0),  # DRC
                (  172,  0.5),  # Equatorial Guinea
                (  173,  0.5),  # Gabon
                (  181,  2.0),  # Madagascar
                (  184, 0.25),  # Mozambique
                (  185,  0.5),  # Rwanda
                (  194, 0.75),  # Lesotho
                (  198, 1.25),  # Zimbabwe
                (  200,  0.5),  # Benin
                (  204, 0.75),  # Chad
                (  210,  2.5),  # Liberia
            ],
            'omicron': [
                (   33, 1.25),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   35,  2.5),  # Georgia
                (   36, 0.75),  # Kazakhstan
                (   37,  6.0),  # Kyrgyzstan
                (   38,  3.0),  # Mongolia
                (   41,  0.4),  # Uzbekistan
                (   43,  4.0),  # Albania
                (   46,  1.5),  # Croatia
                (   50,  3.0),  # Montenegro
                (   51,  2.0),  # Poland
                (   53,  1.5),  # Serbia
                (   54,  0.5),  # Slovakia
                (   55, 0.75),  # Slovenia
                (   58,  2.0),  # Estonia
                (   59,  3.0),  # Latvia
                (   60,  3.0),  # Lithuania
                (   61,  2.0),  # Moldova
                (   62,  5.0),  # Russia
                (   71,  2.0),  # Australia
                (   72,  2.0),  # New Zealand
                (   68,  3.0),  # Republic of Korea
                (43858,  0.5),  # Alberta
                (43859, 0.75),  # British Columbia
                (43860,  2.5),  # Manitoba
                (43866, 0.75),  # Ontario
                (43868, 0.75),  # Quebec
                (43869, 0.75),  # Saskatchewan
                (  530, 0.75),  # Delaware
                (  531,  2.0),  # District of Columbia
                (  536,  1.5),  # Illinois
                (  537, 0.75),  # Indiana
                (  540, 1.25),  # Kentucky
                (  541, 0.75),  # Louisiana
                (  542,  0.5),  # Maine
                (  552, 0.75),  # New Hampshire
                (  560,  0.5),  # Oregon
                (  566,  1.5),  # Texas
                (  568,  0.5),  # Vermont
                (60886,  0.5),  # King and Snohomish Counties
                (  571, 0.75),  # West Virginia
                (   97,  0.6),  # Argentina
                (   98,  0.8),  # Chile
                (   99,  0.8),  # Uruguay
                (   74,  2.0),  # Andorra
                (   76,  1.5),  # Belgium
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
                (  367,  2.0),  # Monaco
                (   90,  1.5),  # Norway
                (   91,  1.5),  # Portugal
                (   93,  0.5),  # Sweden
                (   94,  1.5),  # Switzerland
                (  396,  5.0),  # San Marino
                (60357,  0.5),  # Andalucia
                (60358,  1.5),  # Aragon
                (60365,  1.5),  # Asturias
                (60363,  2.0),  # Balearic Islands
                (60367,  1.5),  # Canary Islands
                (60359,  0.5),  # Cantabria
                (60366, 0.75),  # Murcia
                (60370,  2.0),  # Navarre
                ( 4749,  0.8),  # England
                (  433,  0.6),  # Northern Ireland
                ( 4636,  0.5),  # Wales
                (  105,  0.5),  # Antigua and Barbuda
                (  106,  0.4),  # Bahamas
                (  107,  0.5),  # Barbados
                (  108, 0.75),  # Belize
                (  305,  0.4),  # Bermuda
                (  109,  0.1),  # Cuba
                (  110, 0.25),  # Dominica
                (  111,  0.4),  # Dominican Republic
                (  112,  0.5),  # Grenada
                (  113,  0.4),  # Guyana
                (  114, 0.01),  # Haiti
                (  115,  0.1),  # Jamaica
                (  385, 0.75),  # Puerto Rico
                (  119, 0.75),  # Trinidad and Tobago
                (  121,  4.0),  # Bolivia
                (  122,  1.5),  # Ecuador
                (  125,  0.5),  # Colombia
                (  127,  0.4),  # El Salvador
                (  128,  0.8),  # Guatemala
                (  129, 0.25),  # Honduras
                ( 4644,  2.0),  # Baja California
                ( 4645, 1.25),  # Baja California Sur
                ( 4646, 0.75),  # Campeche
                ( 4649, 0.75),  # Chiapas
                ( 4650,  0.5),  # Chihuahua
                ( 4647,  2.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4654,  0.4),  # Guerrero
                ( 4656, 0.75),  # Jalisco
                ( 4651,  1.5),  # Mexico city
                ( 4658,  0.2),  # Michoacan de Ocampo
                ( 4659,  0.5),  # Morelos
                ( 4662,  0.6),  # Oaxaca
                ( 4664,  0.6),  # Queretaro
                ( 4665,  4.0),  # Quintana Roo
                ( 4667,  1.5),  # Sinaloa
                ( 4669,  3.0),  # Tabasco
                ( 4671, 1.25),  # Tlaxcala
                ( 4673,  5.0),  # Yucatan
                ( 4674,  0.6),  # Zacatecas
                ( 4750,  0.8),  # Acre
                ( 4751,  0.6),  # Alagoas
                ( 4753,  1.8),  # Amapa
                ( 4752,  0.1),  # Amazonas
                ( 4754,  0.5),  # Bahia
                ( 4755, 0.25),  # Ceara
                ( 4756,  1.6),  # Distrito Federal
                ( 4757, 10.0),  # Espirito Santo
                ( 4759, 0.05),  # Maranhao
                ( 4762,  4.0),  # Mato Grosso
                ( 4761,  4.0),  # Mato Grosso do Sul
                ( 4760,  2.0),  # Minas Gerais
                ( 4763,  0.6),  # Para
                ( 4764,  2.0),  # Paraiba
                ( 4765,  0.8),  # Parana
                ( 4766,  0.6),  # Pernambuco
                ( 4767, 0.25),  # Piaui
                ( 4768,  2.0),  # Rio de Janeiro
                ( 4769,  0.6),  # Rio Grande do Norte
                ( 4771,  1.5),  # Roraima
                ( 4775, 0.75),  # Sao Paolo
                ( 4774,  0.1),  # Sergipe
                ( 4776, 0.75),  # Tocantins
                (  136,  0.4),  # Paraguay
                (  160, 0.25),  # Afghanistan
                (  139,0.005),  # Algeria
                (  140,  1.5),  # Bahrain
                (  141,  0.5),  # Egypt
                (  142,  0.3),  # Iran
                (  143,  0.3),  # Iraq
                (  144,  2.0),  # Jordan
                (  145, 0.75),  # Kuwait
                (  146, 0.75),  # Lebanon
                (  147,  1.1),  # Libya
                (  148,  0.5),  # Morocco
                (  150,  0.5),  # Oman
                (  151, 0.03),  # Qatar
                (  152, 0.03),  # Saudi Arabia
                (  522, 0.02),  # Sudan
                (  153,0.005),  # Syrian Arab Republic
                (  154, 0.75),  # Tunisia
                (  155,  1.5),  # Turkey
                (  156, 0.15),  # United Arab Emirates
                ( 157,0.0015),  # Yemen
                (  161, 1.25),  # Bangladesh
                ( 4841,  0.4),  # Andhra Pradesh
                ( 4842,  1.5),  # Arunachal Pradesh
                ( 4843,  0.4),  # Assam
                ( 4844,  0.5),  # Bihar
                ( 4846,  0.5),  # Chhattisgarh
                ( 4849,  3.0),  # Delhi
                ( 4850,  0.4),  # Goa
                ( 4851,  1.5),  # Gujarat
                ( 4855,  0.5),  # Jharkhand
                ( 4856, 0.15),  # Karnataka
                ( 4857,  0.2),  # Kerala
                ( 4860,  0.1),  # Maharashtra
                ( 4861,  1.6),  # Manipur
                ( 4862,  0.8),  # Meghalaya
                ( 4863, 70.0),  # Mizoram
                ( 4864,  0.3),  # Nagaland
                ( 4865, 0.75),  # Odisha
                ( 4845,  5.0),  # Chandigarh
                ( 4858,  5.0),  # Lakshadweep
                ( 4866, 12.0),  # Puducherry
                ( 4867,  0.5),  # Punjab
                ( 4869,  3.0),  # Sikkim
                ( 4870,  0.2),  # Tamil Nadu
                ( 4871,  0.5),  # Telengana
                ( 4873,  0.3),  # Uttar Pradesh
                ( 4875,  0.3),  # West Bengal
                (  164,  0.5),  # Nepal
                (53615,  0.8),  # Azad Jammu & Kashmir
                (53616,  0.2),  # Balochistan
                (53617,  0.4),  # Gilgit-Baltistan
                (53618,  0.6),  # Islamabad Capital Territory
                (53619, 1.75),  # Khyber Pakhtunkhwa
                (53620,  0.3),  # Punjab
                (53621,  0.6),  # Sindh
                (   22,  0.5),  # Fiji
                (  351,  5.0),  # Guam
                (   26, 0.25),  # Papua New Guinea
                (   10,  0.1),  # Cambodia
                (   14,  1.5),  # Maldives
                (   15,  0.1),  # Myanmar
                (   17,  0.1),  # Sri Lanka
                (  186,  3.0),  # Seychelles
                (   19,  0.1),  # Timor-Leste
                (   20,  6.0),  # Viet Nam
                (  169, 1.25),  # Central African Republic (1.5)
                (  170,  0.3),  # Congo
                (  171,  4.0),  # DRC
                (  172, 0.15),  # Equatorial Guinea
                (  173,  0.2),  # Gabon
                (  177, 0.25),  # Djibouti
                (  178, 0.25),  # Eritrea (0.4)
                (  180,  0.8),  # Kenya
                (  184, 0.02),  # Mozambique
                (  185, 0.15),  # Rwanda
                (  187, 0.25),  # Somalia
                (  435,  0.5),  # South Sudan
                (  190, 0.25),  # Uganda
                (  193, 0.75),  # Botswana (X)
                (  197, 0.75),  # Eswatini
                (  194, 0.25),  # Lesotho
                (  195,  0.5),  # Namibia
                (  198,  2.0),  # Zimbabwe
                (  200, 0.02),  # Benin
                (  201,  0.5),  # Burkina Faso
                (  203, 1.25),  # Cabo Verde
                (  202, 0.25),  # Cameroon
                (  204,  0.1),  # Chad
                (  205,  0.5),  # Cote d'Ivoire
                (  206,  0.5),  # Gambia
                (  207,  0.5),  # Ghana
                (  208,  0.5),  # Guinea
                (  209, 0.25),  # Guinea-Bissau
                (  210,  0.8),  # Liberia
                (  211,  2.0),  # Mali
                (  212,  0.5),  # Mauritania
                (  213,  0.3),  # Niger
                (  214,  0.3),  # Nigeria
                (  216,  0.3),  # Senegal
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
                ( 4653,  1.5),  # Guanajuato
                ( 4754,  0.5),  # Bahia
            ],
            'alpha': [
                (   51,  0.5),  # Poland
                (43866, 0.75),  # Ontario
                (  544,  2.0),  # Massachusetts
                (   78, 0.75),  # Denmark
                (  150, 0.75),  # Oman
            ],
            'gamma': [
                ( 4752,  1.5),  # Amazonas
                ( 4754,  0.6),  # Bahia
                ( 4755,  0.6),  # Ceara
                ( 4759,  2.0),  # Maranhao
            ],
            'delta': [
                (   51,  0.5),  # Poland
                (43866, 0.75),  # Ontario
                (  544,  2.0),  # Massachusetts
                (   78, 0.75),  # Denmark
                ( 4647,  2.0),  # Coahuila
                ( 4652,  2.0),  # Durango
                ( 4653,  2.0),  # Guanajuato
                ( 4673,  2.0),  # Yucatan
                ( 4752,  1.5),  # Amazonas
                ( 4754,  0.6),  # Bahia
                ( 4755,  0.6),  # Ceara
                ( 4759,  2.0),  # Maranhao
                (  150, 0.75),  # Oman
            ],
            'omicron': [
                (   46,  2.0),  # Croatia
                (   47,  4.0),  # Czechia
                (   51,  0.5),  # Poland
                (   58,  1.5),  # Estonia
                (43858,  2.0),  # Alberta
                (43859,  3.0),  # British Columbia
                (43860,  6.0),  # Manitoba
                (43866, 0.75),  # Ontario
                (  526,  1.5),  # Arkansas
                (  528,  1.5),  # Colorado
                (  529,  1.5),  # Connecticut
                (  530,  1.5),  # Delaware
                (  531,  1.5),  # District of Columbia
                (  532,  2.0),  # Florida
                (  533, 1.25),  # Georgia
                (  534,  0.5),  # Hawaii
                (  535,  1.5),  # Idaho
                (  536,  2.0),  # Illinois
                (  538,  2.0),  # Iowa
                (  541, 1.25),  # Louisiana
                (  544,  2.0),  # Massachusetts
                (  545, 1.25),  # Michigan
                (  546, 1.25),  # Minnesota
                (  548,  1.5),  # Missouri
                (  550,  1.5),  # Nebraska
                (  551,  1.5),  # Nevada
                (  552, 1.25),  # New Hampshire
                (  553,  1.5),  # New Jersey
                (  555, 1.25),  # New York
                (  556,  1.5),  # North Carolina
                (  558,  2.0),  # Ohio
                (  560,  1.5),  # Oregon
                (  559,  1.5),  # Oklahoma
                (  563,  1.5),  # South Carolina
                (  564,  1.5),  # South Dakota
                (  565,  1.5),  # Tennessee
                (  566, 1.75),  # Texas
                (  567,  2.0),  # Utah
                (  568, 0.75),  # Vermont
                (60886,  0.5),  # King and Snohomish Counties
                (  571, 1.25),  # West Virginia
                (   98,  0.5),  # Chile
                (   78,  2.0),  # Denmark
                (   84,  2.0),  # Ireland
                (   87,  1.5),  # Luxembourg
                (   90, 0.75),  # Norway
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
                (   94,  0.5),  # Switzerland
                ( 4749, 1.25),  # England
                (  433,  0.6),  # Northern Ireland
                (  434,  1.2),  # Scotland
                (  385, 0.75),  # Puerto Rico
                (  126,  3.0),  # Costa Rica
                ( 4643, 0.75),  # Aguascalientes
                ( 4644,  2.0),  # Baja California
                ( 4645, 1.75),  # Baja California Sur
                ( 4646,  0.5),  # Campeche
                ( 4755,  0.5),  # Ceara
                ( 4649,  0.4),  # Chiapas
                ( 4650,  0.4),  # Chihuahua
                ( 4647,  1.5),  # Coahuila
                ( 4648,  0.4),  # Colima
                ( 4652,  3.5),  # Durango
                ( 4653,  3.5),  # Guanajuato
                ( 4654,  0.5),  # Guerrero
                ( 4655, 1.75),  # Hidalgo
                ( 4657, 0.75),  # Mexico state
                ( 4651, 1.25),  # Mexico city
                ( 4658,  0.6),  # Michoacan de Ocampo
                ( 4659,  2.0),  # Morelos
                ( 4661,  2.0),  # Nuevo Leone
                ( 4660, 0.75),  # Nayarit
                ( 4662,  0.6),  # Oaxaca
                ( 4664,  0.3),  # Queretaro
                ( 4665,  2.0),  # Quintana Roo
                ( 4666, 0.75),  # San Luis Potosi
                ( 4669,  2.0),  # Tabasco
                ( 4670,  2.0),  # Tamaulipas
                ( 4673,  2.5),  # Yucatan
                ( 4674, 0.75),  # Zacatecas
                ( 4750,  0.8),  # Acre
                ( 4751,  0.4),  # Alagoas
                ( 4753,  0.6),  # Amapa
                ( 4754,  0.4),  # Bahia
                ( 4756,  1.2),  # Distrito Federal
                ( 4757,  0.4),  # Espirito Santo
                ( 4758,  0.8),  # Goias
                ( 4759,  1.4),  # Maranhao
                ( 4762,  0.6),  # Mato Grosso
                ( 4760,  0.6),  # Minas Gerais
                ( 4763,  0.2),  # Para
                ( 4764,  0.4),  # Paraiba
                ( 4766,  0.1),  # Pernambuco
                ( 4767, 0.25),  # Piaui
                ( 4769,  0.3),  # Rio Grande do Norte
                ( 4772,  0.3),  # Rio Grande do Sul
                ( 4768,  0.6),  # Rio de Janeiro
                ( 4770,  0.6),  # Rondonia
                ( 4771,  0.4),  # Roraima
                ( 4773,  0.8),  # Santa Catarina
                ( 4775,  0.8),  # Sao Paolo
                ( 4774, 0.25),  # Sergipe
                ( 4776,  0.4),  # Tocantins
                (  151, 1.25),  # Qatar
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
                (   34,  2.0),  # Azerbajian
                (   35,  1.5),  # Georgia
                (   37,  3.0),  # Kyrgyzstan
                (  121,  3.0),  # Bolivia
                (  125, 0.75),  # Colombia
                (  126,  0.5),  # Costa Rica
                ( 4647,  2.0),  # Coahuila
                (  133, 0.25),  # Venezuela
                (  155,  0.3),  # Turkey
                (  161,  0.5),  # Bangladesh
                ( 4866,  1.5),  # Puducherry
                (  168,  2.0),  # Angola
                (  169,  2.0),  # Central African Republic
                (  197,  1.5),  # Eswatini
                (  206,  1.5),  # Gambia
            ],
            'other': [
                (  125, 0.75),  # Colombia
            ],
            'alpha': [
                (   34,  3.0),  # Azerbaijan
                (   37,  2.0),  # Kyrgyzstan
                (   43,  5.0),  # Albania
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                (   55, 0.75),  # Slovenia
                (43866, 0.75),  # Ontario
                (43868, 0.75),  # Quebec
                (  562,  0.5),  # Rhode Island
                (   77, 0.75),  # Cyprus
                (   78, 0.75),  # Denmark
                (  121,  3.0),  # Bolivia
                (  160,  3.0),  # Afghanistan
                (  139,  0.5),  # Algeria
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  146, 0.75),  # Lebanon
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
                (  197,  6.0),  # Eswatini
                (  194,  6.0),  # Lesotho
            ],
            'gamma': [
                (   77, 0.75),  # Cyprus
                (  121,  3.0),  # Bolivia
                (  133, 0.25),  # Venezuela
                ( 4753,  2.0),  # Amapa
                ( 4752,  1.5),  # Amazonas
                ( 4751,  0.8),  # Alagoas
                ( 4757,  2.0),  # Espirito Santo
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                ( 4759,  1.5),  # Maranhao
                ( 4762,  1.5),  # Mato Grosso
            ],
            'delta': [
                (   33,  3.0),  # Armenia
                (   34,  4.0),  # Azerbaijan
                (   35,  4.0),  # Georgia
                (   36, 10.0),  # Kazakhstan
                (   37,  2.0),  # Kyrgyzstan
                (   38,  2.0),  # Mongolia
                (   41,  2.0),  # Uzbekistan
                (   43,  5.0),  # Albania
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                (   55, 0.75),  # Slovenia
                (   57,  5.0),  # Belarus
                (   59,  2.0),  # Latvia
                (   60,  2.0),  # Lithuania
                (   63,  2.0),  # Ukraine
                (43866, 0.75),  # Ontario
                (43868, 0.75),  # Quebec
                (  562,  0.5),  # Rhode Island
                (   77, 0.75),  # Cyprus
                (   78, 0.75),  # Denmark
                (   84,  0.6),  # Ireland
                ( 4749,  0.5),  # England
                (  433,  0.5),  # Northern Ireland
                (  434,  0.5),  # Scotland
                ( 4636,  0.5),  # Wales
                (  112,  3.0),  # Grenada
                (  113,  2.0),  # Guyana
                (  118,  2.0),  # Suriname
                (  119,  1.5),  # Trinidad and Tobago
                (  121,  5.0),  # Bolivia
                (  125, 0.75),  # Colombia
                (  126,  0.5),  # Costa Rica
                (  129,  2.0),  # Honduras
                ( 4644,  2.0),  # Baja California
                ( 4647,  2.0),  # Coahuila
                ( 4655,  2.0),  # Hidalgo
                ( 4651,  2.0),  # Mexico city
                ( 4673,  2.0),  # Yucatan
                (  133, 0.25),  # Venezuela
                ( 4751,  0.8),  # Alagoas
                ( 4753,  2.0),  # Amapa
                ( 4752, 1.25),  # Amazonas
                ( 4755,  1.5),  # Ceara
                ( 4757,  2.0),  # Espirito Santo
                ( 4759,  1.4),  # Maranhao
                ( 4762,  1.5),  # Mato Grosso
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                (  136,  1.5),  # Paraguay
                (  160,  3.0),  # Afghanistan
                (  139,  0.5),  # Algeria
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  146, 0.75),  # Lebanon
                (  147,  2.0),  # Libya
                (  148,  2.0),  # Morocco
                (  151,  2.0),  # Qatar
                (  153,  0.5),  # Syrian Arab Republic
                ( 4843,  2.0),  # Assam
                ( 4844,  0.8),  # Bihar
                ( 4846,  0.8),  # Chhattisgarh
                ( 4849,  1.5),  # Delhi
                ( 4851,  0.5),  # Gujarat
                ( 4855,  0.8),  # Jharkhand
                ( 4859,  0.4),  # Madhya Pradesh
                ( 4860,  0.5),  # Maharashtra
                ( 4861,  2.0),  # Manipur
                ( 4862,  2.0),  # Meghalaya
                ( 4863,  2.0),  # Mizoram
                ( 4864,  2.0),  # Nagaland
                ( 4845,  3.0),  # Chandigarh
                ( 4866,  3.0),  # Puducherry
                ( 4869, 0.75),  # Sikkim
                ( 4870,  0.8),  # Tamil Nadu
                ( 4873,  0.4),  # Uttar Pradesh
                ( 4874,  1.5),  # Uttarakhand
                ( 4875,  0.8),  # West Bengal
                (53617,  2.0),  # Gilgit-Baltistan
                (53619,  6.0),  # Khyber Pakhtunkhwa
                (53620,  1.5),  # Punjab
                (53621,  2.0),  # Sindh
                (   15, 0.75),  # Myanmar
                (   19, 0.75),  # Timor-Leste
                (  168,  2.0),  # Angola
                (  169,  2.0),  # Central African Republic
                (  170,  2.0),  # Congo
                (  173,  3.0),  # Gabon
                (  180,  1.5),  # Kenya
                (  181,  3.0),  # Madagascar
                (  184,  2.0),  # Mozambique
                (  187,  3.0),  # Somalia
                (  191,  5.0),  # Zambia
                (  193,  3.0),  # Botswana
                (  197,  5.0),  # Eswatini
                (  194,  4.0),  # Lesotho
                (  195,  3.0),  # Namibia
                (  198, 10.0),  # Zimbabwe
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
                (   34, 12.0),  # Azerbaijan
                (   35, 24.0),  # Georgia
                (   36, 10.0),  # Kazakhstan
                (   37,  3.0),  # Kyrgyzstan
                (   38,  1.5),  # Mongolia
                (   41,  2.5),  # Uzbekistan
                (   43,  8.0),  # Albania
                (   44,  8.0),  # Bosnia and Herzegovina
                (   45,  8.0),  # Bulgaria
                (   46,  4.0),  # Croatia
                (   47,  3.0),  # Czechia
                (   48,  4.0),  # Hungary
                (   49, 18.0),  # North Macedonia
                (   51,  3.0),  # Poland
                (   52,  4.0),  # Romania
                (   53,  5.0),  # Serbia
                (   54,  5.0),  # Slovakia
                (   55, 0.75),  # Slovenia
                (   57, 15.0),  # Belarus
                (   58,  1.5),  # Estonia
                (   59,  6.0),  # Latvia
                (   60,  6.0),  # Lithuania
                (   61,  3.0),  # Moldova
                (   68,  1.5),  # Republic of Korea
                (   69,  0.5),  # Singapore
                (43858,  1.5),  # Alberta
                (43859,  2.0),  # British Columbia
                (43860,  4.0),  # Manitoba
                (43861,  2.0),  # New Brunswick
                (43862,  4.0),  # Newfoundland and Labrador
                (43864,  2.0),  # Nova Scotia
                (43866, 0.75),  # Ontario
                (43868, 0.75),  # Quebec
                (43869,  1.5),  # Saskatchewan
                (  523,  2.0),  # Alabama
                (  524,  2.0),  # Alaska
                (  525,  2.0),  # Arizona
                (  526,  2.0),  # Arkansas
                (  528, 1.75),  # Colorado
                (  530,  1.5),  # Delaware
                (  531, 0.75),  # District of Columbia
                (  532,  2.0),  # Florida
                (  533,  2.0),  # Georgia
                (  535,  2.5),  # Idaho
                (  536,  2.5),  # Illinois
                (  537,  1.5),  # Indiana
                (  538,  2.0),  # Iowa
                (  539,  2.0),  # Kansas
                (  540,  2.5),  # Kentucky
                (  541,  1.5),  # Louisiana
                (  542,  1.5),  # Maine
                (  543,  1.5),  # Maryland
                (  545,  1.5),  # Michigan
                (  546,  1.5),  # Minnesota
                (  547,  1.5),  # Mississippi
                (  548,  2.0),  # Missouri
                (  549,  1.5),  # Montana
                (  550,  2.5),  # Nebraska
                (  551,  2.5),  # Nevada
                (  553,  1.5),  # New Jersey
                (  554,  2.0),  # New Mexico
                (  555, 1.25),  # New York
                (  556,  2.0),  # North Carolina
                (  558,  1.5),  # Ohio
                (  559,  2.5),  # Oklahoma
                (  561,  1.5),  # Pennsylvania
                (  562, 0.75),  # Rhode Island
                (  563, 1.75),  # South Carolina
                (  564,  1.5),  # South Dakota
                (  565,  2.5),  # Tennessee
                (  566,  1.5),  # Texas
                (  567,  2.0),  # Utah
                (  568, 0.75),  # Vermont
                (  569,  1.5),  # Virginia
                (60886, 0.75),  # King and Snohomish Counties
                (  571, 1.75),  # West Virginia
                (  572,  2.5),  # Wisconsin
                (  573,  2.5),  # Wyoming
                (   97,  0.6),  # Argentina
                (   98,  0.8),  # Chile
                (   99,  0.8),  # Uruguay
                (   77, 0.75),  # Cyprus
                (   78,  2.0),  # Denmark
                (   79,  3.0),  # Finland
                (   82, 1.25),  # Greece
                (   84,  0.8),  # Ireland
                (   85,  2.0),  # Israel
                (35512,  2.0),  # Calabria
                (35513,  2.0),  # Sicilia
                (  367, 0.75),  # Monaco
                (   88, 1.25),  # Malta
                (   89,  0.5),  # Netherlands
                (   90,  2.0),  # Norway
                (   91, 1.25),  # Portugal
                (  396,  0.5),  # San Marino
                (60357,  0.5),  # Andalucia
                (60365,  0.5),  # Asturias
                (60364,  1.5),  # Canary Islands
                (60376,  0.5),  # La Rioja
                (60373,  2.0),  # Melilla
                (60366, 0.75),  # Murcia
                (60371, 0.75),  # Valencian Community
                (   94, 0.25),  # Switzerland
                ( 4749,  0.5),  # England
                (  433,  0.4),  # Northern Ireland
                (  434,  0.6),  # Scotland
                ( 4636,  0.4),  # Wales
                (  121, 0.75),  # Bolivia
                (  122,  1.5),  # Ecuador
                (  123, 1.25),  # Peru
                (  105, 0.75),  # Antigua and Barbuda
                (  106, 1.25),  # Bahamas
                (  107,  2.0),  # Barbados
                (  108, 1.75),  # Belize
                (  305,  1.5),  # Bermuda
                (  109, 0.25),  # Cuba
                (  111,  0.3),  # Dominican Republic
                (  112,  4.0),  # Grenada
                (  113,  3.0),  # Guyana
                (  114, 1.25),  # Haiti
                (  115, 1.25),  # Jamaica
                (  385, 0.75),  # Puerto Rico
                (  118,  2.0),  # Suriname
                (  119,  2.5),  # Trinidad and Tobago
                (  422,  2.0),  # US Virgin Islands
                (  121,  7.0),  # Bolivia
                (  125, 0.75),  # Colombia
                (  126, 0.75),  # Costa Rica
                (  127, 1.25),  # El Salvador
                (  129,  3.0),  # Honduras
                ( 4643,  1.5),  # Aguascalientes
                ( 4644,  5.0),  # Baja California
                ( 4645,  3.0),  # Baja California do Sur
                ( 4649,  0.5),  # Chiapas
                ( 4647,  4.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4656,  2.5),  # Jalisco
                ( 4657,  1.5),  # Mexico state
                ( 4651,  2.5),  # Mexico city
                ( 4655,  4.0),  # Hidalgo
                ( 4658,  1.2),  # Michoacan de Ocampo
                ( 4659,  3.0),  # Morelos
                ( 4660,  1.5),  # Nayarit
                ( 4661,  3.5),  # Nuevo Leone
                ( 4663,  2.0),  # Puebla
                ( 4665,  3.0),  # Quintana Roo
                ( 4666,  1.5),  # San Luis Potosi
                ( 4667,  2.0),  # Sinaloa
                ( 4668,  2.5),  # Sonora
                ( 4669,  2.5),  # Tabasco
                ( 4670,  2.5),  # Tamaulipas
                ( 4671,  1.5),  # Tlaxcala
                ( 4672,  1.5),  # Veracruz de Ignacio de la Llave
                ( 4673, 2.75),  # Yucatan
                ( 4674, 1.25),  # Zacatecas
                (  133, 0.25),  # Venezuela
                ( 4750, 1.75),  # Acre
                ( 4753,  1.5),  # Amapa
                ( 4752,  0.8),  # Amazonas
                ( 4754,  1.2),  # Bahia
                ( 4755, 2.75),  # Ceara
                ( 4756,  2.0),  # Distrito Federal
                ( 4757,  2.5),  # Espirito Santo
                ( 4758,  2.5),  # Goias
                ( 4759,  1.6),  # Maranhao
                ( 4762,  1.8),  # Mato Grosso
                ( 4761,  1.8),  # Mato Grosso do Sol
                ( 4760,  1.5),  # Minas Gerais
                ( 4763,  1.2),  # Para
                ( 4764, 0.75),  # Paraiba
                ( 4765,  1.5),  # Parana
                ( 4766,  1.5),  # Pernambuco
                ( 4767,  0.8),  # Piaui
                ( 4769,  0.8),  # Rio Grande do Norte
                ( 4772,  0.6),  # Rio Grande do Sul
                ( 4768,  1.5),  # Rio de Janeiro
                ( 4770,  4.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                ( 4773, 1.25),  # Santa Catarina
                ( 4775,  2.0),  # Sao Paolo
                ( 4774,  0.5),  # Sergipe
                ( 4776,  1.5),  # Tocantins
                (  136,  4.0),  # Paraguay
                (  160,  5.0),  # Afghanistan
                (  139,  0.1),  # Algeria
                (  140,  1.5),  # Bahrain
                (  141,  6.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  144,  3.0),  # Jordan
                (  145, 0.25),  # Kuwait
                (  146,  2.5),  # Lebanon
                (  147,  2.5),  # Libya
                (  148,  2.0),  # Morocco
                (  149,  5.0),  # Palestine
                (  151,  3.0),  # Qatar
                (  152,  0.2),  # Saudi Arabia
                (  522,  2.0),  # Sudan
                (  153, 0.25),  # Syrian Arab Republic
                (  154,  5.0),  # Tunisia
                (  155,  2.0),  # Turkey
                (  156,  0.5),  # United Arab Emirates
                (  157,  2.0),  # Yemen
                (  162,  2.0),  # Bhutan
                ( 4841,  0.4),  # Andhra Pradesh
                ( 4843,  2.0),  # Assam
                ( 4844,  0.4),  # Bihar
                ( 4846,  0.6),  # Chhattisgarh
                ( 4849,  2.0),  # Delhi
                ( 4850,  0.5),  # Goa
                ( 4851,  1.5),  # Gujarat
                ( 4852,  2.0),  # Haryana
                ( 4853,  1.2),  # Himachal Pradesh
                ( 4855,  0.5),  # Jharkhand
                ( 4857,  0.8),  # Kerala
                ( 4859,  0.5),  # Madhya Pradesh
                ( 4860, 0.25),  # Maharashtra
                ( 4861,  3.0),  # Manipur
                ( 4862,  1.5),  # Meghalaya
                ( 4863,  4.0),  # Mizoram
                ( 4864,  2.5),  # Nagaland
                ( 4865,  2.0),  # Odisha
                ( 4845,  5.0),  # Chandigarh
                ( 4866,  2.0),  # Puducherry
                ( 4867,  1.2),  # Punjab
                ( 4868,  1.5),  # Rajasthan
                ( 4869,  1.5),  # Sikkim
                ( 4870,  0.8),  # Tamil Nadu
                ( 4871,  0.4),  # Telengana
                ( 4872,  2.5),  # Tripura
                ( 4873,  0.5),  # Uttar Pradesh
                ( 4874, 1.25),  # Uttarakhand
                ( 4875,  1.5),  # West Bengal
                (  164,  1.2),  # Nepal
                (53615,  3.0),  # Azad Jammu & Kashmir
                (53617,  3.0),  # Gilgit-Baltistan
                (53618,  2.5),  # Islamabad Capital Territory
                (53619,  6.0),  # Khyber Pakhtunkhwa
                (53620, 1.25),  # Punjab
                (53621,  3.0),  # Sindh
                (  354,  2.0),  # Hong Kong Special Administrative Region of China
                (   22,  1.5),  # Fiji
                (  351,  3.0),  # Guam
                (   10,  0.1),  # Cambodia
                (   11,  1.5),  # Indonesia
                (   13,  1.5),  # Malaysia
                (   15,  0.1),  # Myanmar
                (  186,  0.8),  # Seychelles
                (   17, 0.15),  # Sri Lanka
                (   19, 0.25),  # Timor-Leste
                (  168,  3.5),  # Angola
                (  169,  5.0),  # Central African Republic (7.0)
                (  170,  2.5),  # Congo
                (  171,  2.5),  # DRC
                (  172,  3.0),  # Equatorial Guinea
                (  173,  6.0),  # Gabon
                (  178,  2.0),  # Eritrea
                (  179,  2.0),  # Ethiopia
                (  180,  2.5),  # Kenya
                (  181, 10.0),  # Madagascar
                (  182,  4.0),  # Malawi
                (  184,  1.5),  # Mozambique
                (  185,  1.5),  # Rwanda
                (  187,  2.0),  # Somalia
                (  190,  2.0),  # Uganda
                (  191,  5.0),  # Zambia
                (  193,  5.0),  # Botswana (7.0)
                (  197,  7.0),  # Eswatini
                (  194,  2.0),  # Lesotho
                (  195,  7.0),  # Namibia
                (  196,  3.0),  # South Africa
                (  198,  7.0),  # Zimbabwe
                (  200, 0.25),  # Benin
                (  201,  8.0),  # Burkina Faso
                (  202,  1.5),  # Cameroon
                (  203,  1.5),  # Cabo Verde
                (  204, 0.75),  # Chad
                (  205,  3.0),  # Cote d'Ivoire
                (  206,  8.0),  # Gambia
                (  207,  2.0),  # Ghana
                (  208,  4.0),  # Guinea
                (  209, 10.0),  # Guinea-Bissau
                (  211,  2.0),  # Mali
                (  212,  3.0),  # Mauritania
                (  213,  1.5),  # Niger
                (  214,  0.5),  # Nigeria
                (  216,  2.5),  # Senegal
                (  217,  0.5),  # Sierra Leone
                (  218,  1.5),  # Togo
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

    sampled_ode_params = adjust_omega_severity(
        sampled_ode_params, rates_parameters
    )

    return sampled_ode_params


def adjust_omega_severity(
    sampled_ode_params: Dict,
    rates_parameters: RatesParameters,
) -> Dict:
    omega_severity = rates_parameters.omega_severity_parameterization
    severity_calc = {
        'delta': lambda m: sampled_ode_params[f'kappa_delta_{m}'],
        'omicron': lambda m: sampled_ode_params[f'kappa_omicron_{m}'],
        'average': lambda m: (sampled_ode_params[f'kappa_delta_{m}']
                              * sampled_ode_params[f'kappa_omicron_{m}']) ** (1 / 2),
    }[omega_severity]
    for measure in ['case', 'admission', 'death']:
        sampled_ode_params[f'kappa_omega_{measure}'] = severity_calc(measure)

    return sampled_ode_params

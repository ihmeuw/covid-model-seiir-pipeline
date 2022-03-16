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
            'alpha': [
                (   33,  3.0),  # Armenia
                (   61,  3.0),  # Moldova
                (  168, 10.0),  # Angola
            ],
            'gamma': [
                # ( 4760,  3.0),  # Minas Gerais
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
            ],
            'delta': [
                (   33,  3.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   38,  2.0),  # Mongolia
                (   43,  3.0),  # Albania
                (   61,  3.0),  # Moldova
                (   62,  2.0),  # Russia
                ( 4644,  2.0),  # Baja California
                ( 4647,  2.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  2.0),  # Tabasco
                ( 4673,  3.0),  # Yucatan
                # ( 4754,  test_scalar),  # Bahia
                # ( 4762,  test_scalar),  # Mato Grosso
                # ( 4760,  test_scalar),  # Minas Gerais
                # ( 4763,  test_scalar),  # Para
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
                (  147,  2.0),  # Libya
                ( 4863,  8.0),  # Mizoram
                ( 4869,  2.0),  # Sikkim
                (53619,  2.0),  # Khyber Pakhtunkhwa
                (  168,  10.0),  # Angola
                (  171,  2.0),  # DRC
                (  181,  2.0),  # Madagascar
            ],
            'omicron': [
                (   33,  3.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   35,  2.0),  # Georgia
                (   38,  2.0),  # Mongolia
                (   43,  3.0),  # Albania
                (   50,  2.0),  # Montenegro
                (   59,  2.0),  # Latvia
                (   60,  2.0),  # Lithuania
                (   61,  3.0),  # Moldova
                (   62, 10.0),  # Russia
                (43860,  2.0),  # Manitoba
                (   74,  2.0),  # Andorra
                (  121,  0.5),  # Bolivia
                (  122,  1.5),  # Ecuador
                ( 4644,  2.0),  # Baja California
                ( 4647,  3.0),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  3.0),  # Tabasco
                ( 4673, 10.0),  # Yucatan
                ( 4753, 10.0),  # Amapa
                # ( 4754,  test_scalar),  # Bahia
                ( 4756,  1.5),  # Distrito Federal
                # ( 4757,  test_scalar),  # Espirito Santo
                ( 4759,  0.8),  # Maranhao
                # ( 4762,  test_scalar),  # Mato Grosso
                # ( 4760, test_scalar),  # Minas Gerais
                # ( 4763, test_scalar),  # Para
                ( 4764,  3.0),  # Paraiba
                ( 4771,  2.0),  # Roraima
                (  147,  2.0),  # Libya
                ( 4863,  10.0),  # Mizoram
                ( 4869,  2.0),  # Sikkim
                (53619,  2.0),  # Khyber Pakhtunkhwa
                (  351,  3.0),  # Guam
                (  168, 10.0),  # Angola
                (  171, 10.0),  # DRC
                (  181,  2.0),  # Madagascar
                (  198,  1.5),  # Zimbabwe
                (  211,  2.0),  # Mali
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
            'omicron': [
                (43860, 3.0),  # Manitoba
                ( 4644, 2.0),  # Baja California
                ( 4647, 3.0),  # Coahuila
                ( 4653, 3.0),  # Guanajuato
                ( 4665, 2.0),  # Quintana Roo
                ( 4669, 2.0),  # Tabasco
                ( 4673, 4.0),  # Yucatan
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
            'alpha': [
                (   34,  3.0),  # Azerbaijan
                (   43,  5.0),  # Albania
                # (   44,  test_scalar),  # Bosnia and Herzegovina
                (   45,  3.0),  # Bulgaria
                (   49,  3.0),  # North Macedonia
                # (   57,  test_scalar),  # Belarus
                # (  121,  test_scalar),  # Bolivia
                (  160,  3.0),  # Afghanistan
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                # (  181,  test_scalar),  # Madagascar
                (  187,  3.0),  # Somalia
                # (  215,  test_scalar),  # Sao Tome and Principe
            ],
            'beta': [
                (  151,  2.0),  # Qatar
                # (  182,  test_scalar),  # Malawi
                (  184,  3.0),  # Mozambique
                (  193,  2.0),  # Botswana
                # (  197,  test_scalar),  # Eswatini
                (  194,  5.0),  # Lesotho
            ],
            'gamma': [
                # (  121,  test_scalar),  # Bolivia
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
            ],
            'delta': [
                (   33,  3.0),  # Armenia
                (   34,  3.0),  # Azerbaijan
                (   35,  3.0),  # Georgia
                # (   36,  test_scalar),  # Kazakhstan
                (   38,  2.0),  # Mongolia
                (   43,  5.0),  # Albania
                # (   44,  test_scalar),  # Bosnia and Herzegovina
                (   45,  3.0),  # Bulgaria
                (   50,  2.0),  # Montenegro
                (   49,  3.0),  # North Macedonia
                # (   57,  test_scalar),  # Belarus
                (   59,  2.0),  # Latvia
                (   60,  2.0),  # Lithuania
                # (  121,  test_scalar),  # Bolivia
                ( 4673,  2.0),  # Yucatan
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                (  160,  3.0),  # Afghanistan
                (  141,  2.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  147,  2.0),  # Libya
                (  148,  2.0),  # Morocco
                (  151,  2.0),  # Qatar
                ( 4861,  2.0),  # Manipur
                ( 4863,  2.0),  # Mizoram
                (53619,  3.0),  # Khyber Pakhtunkhwa
                (53621,  2.0),  # Sindh
                (  168,  3.0),  # Angola
                # (  169,  test_scalar),  # Central African Republic
                # (  181,  test_scalar),  # Madagascar
                (  184,  3.0),  # Mozambique
                (  187,  3.0),  # Somalia
                (  190,  2.0),  # Uganda
                (  191,  3.0),  # Zambia
                (  193,  2.0),  # Botswana
                # (  197,  test_scalar),  # Eswatini
                (  194,  5.0),  # Lesotho
                (  195,  2.0),  # Namibia
                (  198,  8.0),  # Zimbabwe
                (  205,  2.0),  # Cote d'Ivoire
                # (  206,  test_scalar),  # Gambia
                (  209,  3.0),  # Guinea Bissau
                (  210,  3.0),  # Liberia
                # (  215,  test_scalar),  # Sao Tome and Principe
                (  216,  2.0),  # Senegal
            ],
            'omicron': [
                (   33, 10.0),  # Armenia
                (   34, 10.0),  # Azerbaijan
                (   35, 10.0),  # Georgia
                # (   36,  test_scalar),  # Kazakhstan
                (   38,  2.0),  # Mongolia
                (   41,  2.0),  # Uzbekistan
                (   43,  5.0),  # Albania
                # (   44,  test_scalar),  # Bosnia and Herzegovina
                (   45,  5.0),  # Bulgaria
                (   46,  2.0),  # Croatia
                (   47,  2.0),  # Czechia
                (   48,  2.0),  # Hungary
                (   50,  10.0),  # Montenegro
                (   49,  8.0),  # North Macedonia
                (   52,  2.0),  # Romania
                # (   57,  test_scalar),  # Belarus
                (   59,  8.0),  # Latvia
                (   60,  2.0),  # Lithuania
                (   61,  1.5),  # Moldova
                (43860,  2.0),  # Manitoba
                # (  121,  test_scalar),  # Bolivia
                (  122,  2.0),  # Ecuador
                (  113,  2.0),  # Guyana
                (  129,  3.0),  # Honduras
                ( 4644,  3.0),  # Baja California
                # ( 4647,  test_scalar),  # Coahuila
                ( 4653,  2.0),  # Guanajuato
                ( 4665,  2.0),  # Quintana Roo
                ( 4669,  2.0),  # Tabasco
                ( 4673,  4.0),  # Yucatan
                ( 4753,  2.5),  # Amapa
                ( 4756,  2.0),  # Distrito Federal
                ( 4757,  2.0),  # Espirito Santo
                ( 4759,  1.5),  # Maranhao
                ( 4761,  2.0),  # Mato Grosso do Sol
                ( 4770,  2.0),  # Rondonia
                ( 4771,  2.0),  # Roraima
                (  160,  3.0),  # Afghanistan
                (  141,  3.0),  # Egypt
                (  143,  2.0),  # Iraq
                (  147,  2.0),  # Libya
                (  148,  2.0),  # Morocco
                (  151,  2.0),  # Qatar
                (  154,  3.0),  # Tunisia
                ( 4861,  2.5),  # Manipur
                ( 4863,  4.0),  # Mizoram
                (53619,  5.0),  # Khyber Pakhtunkhwa
                (53621,  2.0),  # Sindh
                (  351,  3.0),  # Guam
                (  168,  5.0),  # Angola
                # (  169,  test_scalar),  # Central African Republic
                (  171,  2.5),  # DRC
                (  180,  3.0),  # Kenya
                # (  181,  test_scalar),  # Madagascar
                # (  182,  test_scalar),  # Malawi
                (  184,  3.0),  # Mozambique
                (  187,  5.0),  # Somalia
                (  190,  2.0),  # Uganda
                (  191,  5.0),  # Zambia
                (  193,  2.0),  # Botswana
                # (  197,  test_scalar),  # Eswatini
                (  194,  5.0),  # Lesotho
                (  195, 10.0),  # Namibia
                (  198, 10.0),  # Zimbabwe
                (  201, 10.0),  # Burkina Faso
                (  205,  3.0),  # Cote d'Ivoire
                (  209, 10.0),  # Guinea Bissau
                (  210,  5.0),  # Liberia
                (  211,  2.0),  # Mali
                # (  215,  test_scalar),  # Sao Tome and Principe
                (  216,  2.0),  # Senegal
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


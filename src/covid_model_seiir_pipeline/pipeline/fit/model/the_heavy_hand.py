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
                (   33, 3.0),  # Armenia
                (   61, test_scalar),  # Moldova
                (  168, test_scalar),  # Angola
            ],
            'gamma': [
                ( 4760, test_scalar),  # Minas Gerais
                ( 4764, test_scalar),  # Paraiba
                ( 4771, test_scalar),  # Roraima
            ],
            'delta': [
                (   33, test_scalar),  # Armenia
                (   34, test_scalar),  # Azerbaijan
                (   38, test_scalar),  # Mongolia
                (   43, test_scalar),  # Albania
                (   61, test_scalar),  # Moldova
                (   62, test_scalar),  # Russia
                ( 4644, test_scalar),  # Baja California
                ( 4647, test_scalar),  # Coahuila
                ( 4653, test_scalar),  # Guanajuato
                ( 4665, test_scalar),  # Quintana Roo
                ( 4669, test_scalar),  # Tabasco
                ( 4673, test_scalar),  # Yucatan
                ( 4753, test_scalar),  # Amapa
                ( 4754, test_scalar),  # Bahia
                ( 4762, test_scalar),  # Mato Grosso
                ( 4760, test_scalar),  # Minas Gerais
                ( 4763, test_scalar),  # Para
                ( 4764, test_scalar),  # Paraiba
                ( 4771, test_scalar),  # Roraima
                (  147, test_scalar),  # Libya
                ( 4863, test_scalar),  # Mizoram
                ( 4869, test_scalar),  # Sikkim
                (53619, test_scalar),  # Khyber Pakhtunkhwa
                (  168, test_scalar),  # Angola
                (  171, test_scalar),  # DRC
                (  181, test_scalar),  # Madagascar
            ],
            'omicron': [
                (   33, test_scalar),  # Armenia
                (   34, test_scalar),  # Azerbaijan
                (   35, test_scalar),  # Georgia
                (   38, test_scalar),  # Mongolia
                (   43, test_scalar),  # Albania
                (   50, test_scalar),  # Montenegro
                (   59, test_scalar),  # Latvia
                (   60, test_scalar),  # Lithuania
                (   61, test_scalar),  # Moldova
                (   62, test_scalar),  # Russia
                (43860, test_scalar),  # Manitoba
                (   74, test_scalar),  # Andorra
                (  121, test_scalar),  # Bolivia
                (  122, test_scalar),  # Ecuador
                ( 4644, test_scalar),  # Baja California
                ( 4647, test_scalar),  # Coahuila
                ( 4653, test_scalar),  # Guanajuato
                ( 4665, test_scalar),  # Quintana Roo
                ( 4669, test_scalar),  # Tabasco
                ( 4673, test_scalar),  # Yucatan
                ( 4753, test_scalar),  # Amapa
                ( 4754, test_scalar),  # Bahia
                ( 4756, test_scalar),  # Distrito Federal
                ( 4757, test_scalar),  # Espirito Santo
                ( 4759, test_scalar),  # Maranhao
                ( 4762, test_scalar),  # Mato Grosso
                ( 4760, test_scalar),  # Minas Gerais
                ( 4763, test_scalar),  # Para
                ( 4764, test_scalar),  # Paraiba
                ( 4771, test_scalar),  # Roraima
                (  147, test_scalar),  # Libya
                ( 4863, test_scalar),  # Mizoram
                ( 4869, test_scalar),  # Sikkim
                (53619, test_scalar),  # Khyber Pakhtunkhwa
                (  351, test_scalar),  # Guam
                (  168, test_scalar),  # Angola
                (  171, test_scalar),  # DRC
                (  181, test_scalar),  # Madagascar
                (  198, test_scalar),  # Zimbabwe
                (  211, test_scalar),  # Mali
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
                (   47, test_scalar),  # Armenia
                (43860, test_scalar),  # Manitoba
                ( 4644, test_scalar),  # Baja California
                ( 4647, test_scalar),  # Coahuila
                ( 4653, test_scalar),  # Guanajuato
                ( 4665, test_scalar),  # Quintana Roo
                ( 4669, test_scalar),  # Tabasco
                ( 4673, test_scalar),  # Yucatan
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
                (   34, test_scalar),  # Azerbaijan
                (   43, test_scalar),  # Albania
                (   44, test_scalar),  # Bosnia and Herzegovina
                (   45, test_scalar),  # Bulgaria
                (   49, test_scalar),  # North Macedonia
                (   57, test_scalar),  # Belarus
                (  121, test_scalar),  # Bolivia
                (  160, test_scalar),  # Afghanistan
                (  141, test_scalar),  # Egypt
                (  143, test_scalar),  # Iraq
                (  181, test_scalar),  # Madagascar
                (  187, test_scalar),  # Somalia
                (  215, test_scalar),  # Sao Tome and Principe
            ],
            'beta': [
                (  151, test_scalar),  # Qatar
                (  182, test_scalar),  # Malawi
                (  184, test_scalar),  # Mozambique
                (  193, test_scalar),  # Botswana
                (  197, test_scalar),  # Eswatini
                (  194, test_scalar),  # Lesotho
            ],
            'gamma': [
                (  121, test_scalar),  # Bolivia
                ( 4757, test_scalar),  # Espirito Santo
                ( 4770, test_scalar),  # Rondonia
                ( 4771, test_scalar),  # Roraima
            ],
            'delta': [
                (   33, 3.0),  # Armenia
                (   34, test_scalar),  # Azerbaijan
                (   35, test_scalar),  # Georgia
                (   36, test_scalar),  # Kazakhstan
                (   38, test_scalar),  # Mongolia
                (   41, test_scalar),  # Uzbekistan
                (   43, test_scalar),  # Albania
                (   44, test_scalar),  # Bosnia and Herzegovina
                (   45, test_scalar),  # Bulgaria
                (   50, test_scalar),  # Montenegro
                (   49, test_scalar),  # North Macedonia
                (   57, test_scalar),  # Belarus
                (   59, test_scalar),  # Latvia
                (   60, test_scalar),  # Lithuania
                (  121, test_scalar),  # Bolivia
                ( 4757, test_scalar),  # Espirito Santo
                ( 4770, test_scalar),  # Rondonia
                ( 4771, test_scalar),  # Roraima
                (  160, test_scalar),  # Afghanistan
                (  141, test_scalar),  # Egypt
                (  143, test_scalar),  # Iraq
                (  147, test_scalar),  # Libya
                (  148, test_scalar),  # Morocco
                (  151, test_scalar),  # Qatar
                ( 4861, test_scalar),  # Manipur
                ( 4863, test_scalar),  # Mizoram
                (53619, test_scalar),  # Khyber Pakhtunkhwa
                (53621, test_scalar),  # Sindh
                (  168, test_scalar),  # Angola
                (  169, test_scalar),  # Central African Republic
                (  181, test_scalar),  # Madagascar
                (  184, test_scalar),  # Mozambique
                (  187, test_scalar),  # Somalia
                (  190, test_scalar),  # Uganda
                (  191, test_scalar),  # Zambia
                (  193, test_scalar),  # Botswana
                (  197, test_scalar),  # Eswatini
                (  194, test_scalar),  # Lesotho
                (  195, test_scalar),  # Namibia
                (  198, test_scalar),  # Zimbabwe
                (  205, test_scalar),  # Cote d'Ivoire
                (  206, test_scalar),  # Gambia
                (  209, test_scalar),  # Guinea Bissau
                (  210, test_scalar),  # Liberia
                (  215, test_scalar),  # Sao Tome and Principe
                (  216, test_scalar),  # Senegal
                (  218, test_scalar),  # Togo
            ],
            'omicron': [
                (   33, 10.0),  # Armenia
                (   34, test_scalar),  # Azerbaijan
                (   35, test_scalar),  # Georgia
                (   36, test_scalar),  # Kazakhstan
                (   38, test_scalar),  # Mongolia
                (   41, test_scalar),  # Uzbekistan
                (   43, test_scalar),  # Albania
                (   44, test_scalar),  # Bosnia and Herzegovina
                (   45, test_scalar),  # Bulgaria
                (   46, test_scalar),  # Croatia
                (   47, test_scalar),  # Czechia
                (   48, test_scalar),  # Hungary
                (   50, test_scalar),  # Montenegro
                (   49, test_scalar),  # North Macedonia
                (   52, test_scalar),  # Romania
                (   57, test_scalar),  # Belarus
                (   59, test_scalar),  # Latvia
                (   60, test_scalar),  # Lithuania
                (   61, test_scalar),  # Moldova
                (43860, test_scalar),  # Manitoba
                (  121, test_scalar),  # Bolivia
                (  122, test_scalar),  # Ecuador
                (  113, test_scalar),  # Guyana
                (  129, test_scalar),  # Honduras
                ( 4644, test_scalar),  # Baja California
                ( 4647, test_scalar),  # Coahuila
                ( 4653, test_scalar),  # Guanajuato
                ( 4665, test_scalar),  # Quintana Roo
                ( 4669, test_scalar),  # Tabasco
                ( 4673, test_scalar),  # Yucatan
                ( 4753, test_scalar),  # Amapa
                ( 4756, test_scalar),  # Distrito Federal
                ( 4757, test_scalar),  # Espirito Santo
                ( 4759, test_scalar),  # Maranhao
                ( 4761, test_scalar),  # Mato Grosso do Sol
                ( 4770, test_scalar),  # Rondonia
                ( 4771, test_scalar),  # Roraima
                (  160, test_scalar),  # Afghanistan
                (  141, test_scalar),  # Egypt
                (  143, test_scalar),  # Iraq
                (  147, test_scalar),  # Libya
                (  148, test_scalar),  # Morocco
                (  151, test_scalar),  # Qatar
                (  154, test_scalar),  # Tunisia
                ( 4861, test_scalar),  # Manipur
                ( 4863, test_scalar),  # Mizoram
                (53617, test_scalar),  # Gilgit-Baltistan
                (53619, test_scalar),  # Khyber Pakhtunkhwa
                (53621, test_scalar),  # Sindh
                (  351, test_scalar),  # Guam
                (  168, test_scalar),  # Angola
                (  169, test_scalar),  # Central African Republic
                (  171, test_scalar),  # DRC
                (  180, test_scalar),  # Kenya
                (  181, test_scalar),  # Madagascar
                (  182, test_scalar),  # Malawi
                (  184, test_scalar),  # Mozambique
                (  187, test_scalar),  # Somalia
                (  190, test_scalar),  # Uganda
                (  191, test_scalar),  # Zambia
                (  193, test_scalar),  # Botswana
                (  197, test_scalar),  # Eswatini
                (  194, test_scalar),  # Lesotho
                (  195, test_scalar),  # Namibia
                (  198, test_scalar),  # Zimbabwe
                (  201, test_scalar),  # Burkina Faso
                (  205, test_scalar),  # Cote d'Ivoire
                (  209, test_scalar),  # Guinea Bissau
                (  210, test_scalar),  # Liberia
                (  211, test_scalar),  # Mali
                (  215, test_scalar),  # Sao Tome and Principe
                (  216, test_scalar),  # Senegal
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


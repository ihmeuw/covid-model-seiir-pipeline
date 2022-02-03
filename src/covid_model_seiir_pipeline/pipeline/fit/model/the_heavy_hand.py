from typing import Dict

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import sample_parameter


def rescale_kappas(sampled_ode_params: Dict, compartments: pd.DataFrame, draw_id: int):
    delta_infections = compartments.filter(like='Infection_all_delta_all').sum(axis=1).groupby('location_id').max()
    delta_cases = compartments.filter(like='Case_all_delta_all').sum(axis=1).groupby('location_id').max()
    all_infections = compartments.filter(like='Infection_all_all_all').sum(axis=1).groupby('location_id').max()
    all_cases = compartments.filter(like='Case_all_all_all').sum(axis=1).groupby('location_id').max()
    max_idr = 0.9
    p_symptomatic_pre_omicron = 0.5
    p_symptomatic_post_omicron = 1 - sample_parameter('p_asymptomatic_omicron', draw_id=draw_id,
                                                      lower=0.85, upper=0.95)
    minimum_asymptomatic_idr_fraction = 0.1
    maximum_asymptomatic_idr = 0.2

    idr_scaling_factors = [
        # # 0.2
        # (  105, 0.2),  # Antigua and Barbuda
        # (  117, 0.2),  # Saint Vincent and the Grenadines
        # (  156, 0.2),  # United Arab Emirates
        # (   26, 0.2),  # Papua New Guinea
        # (  213, 0.2),  # Niger
        # (  214, 0.2),  # Nigeria
        # (  217, 0.2),  # Sierra Leone
        # # 0.4
        # (  106, 0.4),  # Bahamas
        # (  393, 0.4),  # Saint Kitts and Nevis
        # (  376, 0.4),  # Northern Mariana Islands
        # (  172, 0.4),  # Equatorial Guinea
        # (  173, 0.4),  # Gabon
        # (  175, 0.4),  # Burundi
        # (  185, 0.4),  # Rwanda
        # (  207, 0.4),  # Ghana
        # (  210, 0.4),  # Liberia
        # # 0.6
        # (  170, 0.6),  # Congo
        # (  187, 0.6),  # Somalia
        # (  190, 0.6),  # Uganda
        # (  206, 0.6),  # Gambia
        # (  208, 0.6),  # Guinea
        # (  216, 0.6),  # Senegal
        # (  218, 0.6),  # Togo
        # # 0.8
        # (  179, 0.8),  # Ethiopia
        # (  205, 0.8),  # Côte d'Ivoire
        # # 1.2
        # (  191, 1.2),  # Zambia
        # (  193, 1.2),  # Botswana
        # (  201, 1.2),  # Burkina Faso
        # # 1.4
        # (   49, 1.4),  # North Macedonia
        # (  528, 1.4),  # Colorado
        # (  529, 1.4),  # Connecticut
        # (  531, 1.4),  # District of Columbia
        # (  532, 1.4),  # Florida
        # (  536, 1.4),  # Illinois
        # (  543, 1.4),  # Maryland
        # (  553, 1.4),  # New Jersey
        # (  555, 1.4),  # New York
        # (  558, 1.4),  # Ohio
        # (  180, 1.4),  # Kenya
        # (  182, 1.4),  # Malawi
        # (  198, 1.4),  # Zimbabwe
        # # 1.6
        # (  396, 1.6),  # San Marino
        # (60358, 1.6),  # Aragon
        # (60374, 1.6),  # Basque Country
        # (60359, 1.6),  # Cantabria
        # (60360, 1.6),  # Castilla-La Mancha
        # (60361, 1.6),  # Community of Madrid
        # (60363, 1.6),  # Balearic Islands
        # (60367, 1.6),  # Castile and León
        # (60366, 1.6),  # Murcia
        # (  168, 1.6),  # Angola
        # # 2.0
        # (43860, 2.0),  # Manitoba
        # (   97, 2.0),  # Argentina
        # (  121, 2.0),  # Bolivia
        # (  197, 2.0),  # Eswatini
        # # 3.0
        # (   83, 3.0),  # Iceland
        # (60364, 3.0),  # Canary Islands
        # (60370, 3.0),  # Navarre
        # (60373, 3.0),  # Melilla
        # ( 4849, 3.0),  # Delhi
        # ( 4863, 3.0),  # Mizoram
        # (  196, 3.0),  # South Africa
        # (  203, 3.0),  # Cabo Verde
        # # 10.0
        # (   50, 10.0),  # Montenegro
        # (60376, 10.0),  # La Rioja
        # (  176, 10.0),  # Comoros
        # (  186, 10.0),  # Seychelles
        # (  215, 10.0),  # Sao Tome and Principe
    ]
    # IDR = p_s * IDR_s + p_a * IDR_a
    # IDR_a = (IDR - IDR_s * p_s) / p_a
    # IDR_a >= min_frac_a * IDR
    # IDR_a <= 0.2 [post]
    delta_idr = delta_cases / delta_infections
    delta_idr = delta_idr.fillna(all_cases / all_infections)
    capped_delta_idr = np.minimum(delta_idr, max_idr)
    for location_id, idr_scaling_factor in idr_scaling_factors:
        capped_delta_idr.loc[location_id] *= idr_scaling_factor
    idr_asymptomatic = (capped_delta_idr - max_idr * p_symptomatic_pre_omicron) / (1 - p_symptomatic_pre_omicron)
    idr_asymptomatic = np.maximum(idr_asymptomatic, capped_delta_idr * minimum_asymptomatic_idr_fraction)
    idr_symptomatic = (capped_delta_idr - idr_asymptomatic * (1 - p_symptomatic_pre_omicron)) / p_symptomatic_pre_omicron
    idr_asymptomatic = np.minimum(idr_asymptomatic, maximum_asymptomatic_idr)
    omicron_idr = p_symptomatic_post_omicron * idr_symptomatic + (1 - p_symptomatic_post_omicron) * idr_asymptomatic
    sampled_ode_params['kappa_omicron_case'] = (omicron_idr / delta_idr).rename('kappa_omicron_case')

    ihr_scaling_factors = [
        # # 1.4
        # (  528, 1.4),  # Colorado
        # (  529, 1.4),  # Connecticut
        # (  531, 1.4),  # District of Columbia
        # (  532, 1.4),  # Florida
        # (  536, 1.4),  # Illinois
        # (  543, 1.4),  # Maryland
        # (  553, 1.4),  # New Jersey
        # (  555, 1.4),  # New York
        # (  558, 1.4),  # Ohio
        # # 1.6
        # (60358, 1.6),  # Aragon
        # (60374, 1.6),  # Basque Country
        # (60359, 1.6),  # Cantabria
        # (60360, 1.6),  # Castilla-La Mancha
        # (60361, 1.6),  # Community of Madrid
        # (60363, 1.6),  # Balearic Islands
        # (60367, 1.6),  # Castile and León
        # (60366, 1.6),  # Murcia
        # # 2.0
        # (43860, 2.0),  # Manitoba
        # # 3.0
        # (60364, 3.0),  # Canary Islands
        # (60370, 3.0),  # Navarre
        # (60373, 3.0),  # Melilla
        # (  196, 3.0),  # South Africa
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
        # # 1.6
        # (60358, 1.6),  # Aragon
        # (60374, 1.6),  # Basque Country
        # (60359, 1.6),  # Cantabria
        # (60360, 1.6),  # Castilla-La Mancha
        # (60361, 1.6),  # Community of Madrid
        # (60363, 1.6),  # Balearic Islands
        # (60367, 1.6),  # Castile and León
        # (60366, 1.6),  # Murcia
        # # 3.0
        # (60364, 3.0),  # Canary Islands
        # (60370, 3.0),  # Navarre
        # (60373, 3.0),  # Melilla
        # ( 4849, 3.0),  # Delhi
        # ( 4863, 3.0),  # Mizoram
        # (  196, 3.0),  # South Africa
        # (  203, 3.0),  # Cabo Verde
        # # 10.0
        # (   50, 10.0),  # Montenegro
        # (60376, 10.0),  # La Rioja
        # (  176, 10.0),  # Comoros
        # (  186, 10.0),  # Seychelles
        # (  215, 10.0),  # Sao Tome and Principe
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

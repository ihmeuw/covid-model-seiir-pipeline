from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters

logger = cli_tools.task_performance_logger


def rescale_kappas(
    measure: str,
    location_ids: List[int],
    sampled_ode_params: Dict,
    rates_parameters: RatesParameters,
) -> Dict:
    kappa_scaling_factors_path = Path(__file__).parent / 'kappa_scaling_factors'

    if rates_parameters.manual_fixes:
        manual_scaling_factors = yaml.full_load((kappa_scaling_factors_path / '_manual.yaml').read_text())
        if manual_scaling_factors is None:
            manual_scaling_factors = {}
        else:
            manual_scaling_factors = manual_scaling_factors.get(measure, {})
    else:
        manual_scaling_factors = {}

    for variant in VARIANT_NAMES:
        if rates_parameters.calibration_type:
            try:
                scaling_factors = yaml.full_load((kappa_scaling_factors_path /
                                                  rates_parameters.calibration_type /
                                                  f'{variant}.yaml').read_text())
                scaling_factors = scaling_factors[measure]

                logger.info(f'Applying {variant} kappa scalars to {len(scaling_factors)} locations'
                            f' (calibration type {rates_parameters.calibration_type}')
            except FileNotFoundError:
                logger.warning(f'No kappa scaling factors for {variant}'
                               f' (calibration type {rates_parameters.calibration_type})')
                scaling_factors = {}
        else:
            scaling_factors = {}

        flag_manual = False
        for location_id, factor in manual_scaling_factors.get(variant, {}).items():
            scaling_factors[location_id] = scaling_factors.get(location_id, 1) * factor
            flag_manual = True
        if flag_manual:
            logger.info(f'Including manual kappa scalars for {variant}')

        kappa = pd.Series(
            sampled_ode_params[f'kappa_{variant}_{measure}'],
            index=location_ids,
            name=f'kappa_{variant}_{measure}'
        )
        for location_id, factor in scaling_factors.items():
            try:
                kappa.loc[location_id] *= factor
            except KeyError:
                logger.warning(
                    'Kappa scalar provided for a location not in '
                    f'compartments: {location_id}'
                )
        sampled_ode_params[f'kappa_{variant}_{measure}'] = kappa

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

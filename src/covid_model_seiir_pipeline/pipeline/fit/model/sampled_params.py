from typing import Dict, NamedTuple, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import utilities
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.pipeline.fit.specification import (
    RatesParameters,
    FitParameters,
    DiscreteUniformSampleable,
)


class Durations(NamedTuple):
    exposure_to_case: pd.Series
    exposure_to_admission: pd.Series
    exposure_to_seroconversion: pd.Series
    exposure_to_death: pd.Series
    pcr_to_seropositive: pd.Series
    admission_to_seropositive: pd.Series
    seropositive_to_death: pd.Series
    max_lag: int

    def to_ints(self):
        return Durations(*[x.max() if isinstance(x, pd.Series) else x for x in self])

    def to_dict(self):
        return self.to_ints()._asdict()


def sample_durations(params: RatesParameters, draw_id: int, hierarchy: pd.DataFrame) -> Durations:
    random_state = utilities.get_random_state(f'epi_durations_draw_{draw_id}')

    exposure_to_admission = random_state.choice(_to_range(params.exposure_to_admission))
    exposure_to_seroconversion = random_state.choice(_to_range(params.exposure_to_seroconversion))
    admission_to_death = random_state.choice(_to_range(params.admission_to_death))
    max_lag = max(_to_range(params.exposure_to_admission)) + max(_to_range(params.admission_to_death))

    locations = hierarchy.location_id.tolist()

    return Durations(
        exposure_to_case=pd.Series(exposure_to_admission, index=locations),
        exposure_to_admission=pd.Series(exposure_to_admission, index=locations),
        exposure_to_seroconversion=pd.Series(exposure_to_seroconversion, index=locations),
        exposure_to_death=pd.Series(exposure_to_admission + admission_to_death, index=locations),
        pcr_to_seropositive=pd.Series(exposure_to_seroconversion - exposure_to_admission, index=locations),
        admission_to_seropositive=pd.Series(exposure_to_seroconversion - exposure_to_admission, index=locations),
        seropositive_to_death=pd.Series((exposure_to_admission + admission_to_death) - exposure_to_seroconversion, index=locations),
        max_lag=max_lag,
    )


def _to_range(val: DiscreteUniformSampleable):
    if isinstance(val, int):
        return [val]
    else:
        return list(range(val[0], val[1] + 1))


class VariantRR(NamedTuple):
    ifr: float
    ihr: float
    idr: float
    omicron_ifr: float
    omicron_ihr: float
    omicron_idr: float
    ba5_ifr: float
    ba5_ihr: float
    ba5_idr: float
    omega_ifr: float
    omega_ihr: float
    omega_idr: float


def sample_variant_severity(params: RatesParameters, draw_id: int) -> VariantRR:
    random_state = utilities.get_random_state(f'variant_severity_draw_{draw_id}')

    params = params.to_dict()
    rrs = {}
    for ratio in ['ifr', 'ihr', 'idr']:
        rr_spec = params[f'{ratio}_risk_ratio']
        if isinstance(rr_spec, (int, float)):
            rrs[ratio] = float(rr_spec)
        else:
            mean, lower, upper = rr_spec
            mu = np.log(mean)
            sigma = (np.log(upper) - np.log(lower)) / 3.92
            rrs[ratio] = np.exp(random_state.normal(mu, sigma))

        for variant in ['omicron', 'ba5']:
            variant_spec = params[f'{variant}_{ratio}_scalar']
            if isinstance(variant_spec, (int, float)):
                value = variant_spec
            else:
                value = sample_parameter(f'{variant}_{ratio}_scalar', draw_id, *variant_spec)
            rrs[f'{variant}_{ratio}'] = rrs[ratio] * value

        omega_severity = params['omega_severity_parameterization']
        rrs[f'omega_{ratio}'] = {
            'delta': rrs[ratio],
            'omicron': rrs[f'omicron_{ratio}'],
            'ba5': rrs[f'ba5_{ratio}'],
            'average': (rrs[ratio] * rrs[f'omicron_{ratio}']) ** (1/2),
        }[omega_severity]

    return VariantRR(**rrs)


def sample_day_inflection(params: RatesParameters, draw_id: int) -> pd.Timestamp:
    random_state = utilities.get_random_state(f'day_inflection_draw_{draw_id}')
    return pd.Timestamp(str(random_state.choice(params.day_inflection_options)))


def sample_ode_params(variant_rr: VariantRR,
                      beta_fit_params: FitParameters,
                      draw_id: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    beta_fit_params = beta_fit_params.to_dict()
    sampled_params = {}
    phis = {}
    for parameter, param_spec in beta_fit_params.items():
        if 'phi' in parameter or parameter == 'omega_invasion_date':
            continue

        if isinstance(param_spec, (int, float)):
            value = param_spec
        else:
            value = sample_parameter(parameter, draw_id, *param_spec)

        if 'kappa' in parameter:
            sampled_params[parameter] = value
            variant = parameter.split('_')[1]
            phi_spec = beta_fit_params[parameter.replace('kappa', 'phi')]
            if isinstance(phi_spec, (int, float)):
                phis[variant] = phi_spec
            elif isinstance(param_spec, (tuple, list)):
                x1, x2 = param_spec
                y1, y2 = phi_spec
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                phis[variant] = m * value + b

        key = f'{parameter}_infection'
        sampled_params[key] = value

    s = -12345.0
    phi_matrix = pd.DataFrame(
        data=np.array([
            # TO
            # none ancestral alpha beta gamma delta omicron ba5 other omega    # FROM
            [0.0,       0.0,  0.0, 0.0,  0.0,  0.0,    0.0, 0.0, 0.0,   0.0],  # none
            [1.0,       1.0,    s,   s,    s,    s,      s,   s,   s,     s],  # ancestral
            [1.0,         s,  1.0,   s,    s,    s,      s,   s,   s,     s],  # alpha
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,      s,   s, 1.0,     s],  # beta
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,      s,   s, 1.0,     s],  # gamma
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,      s,   s, 1.0,     s],  # delta
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,    1.0,   s, 1.0,     s],  # omicron
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,    1.0, 1.0, 1.0,     s],  # ba5
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,      s,   s, 1.0,     s],  # other
            [1.0,       1.0,  1.0, 1.0,  1.0,  1.0,    1.0, 1.0, 1.0,   1.0],  # omega
        ]),
        columns=VARIANT_NAMES,
        index=pd.Index(VARIANT_NAMES, name='variant'),
    )
    for variant, phi in phis.items():
        phi_matrix.loc[phi_matrix[variant] == s, variant] = phi

    variant_rr = variant_rr._asdict()
    for measure, rate in (('death', 'ifr'), ('admission', 'ihr'), ('case', 'idr')):
        for variant in VARIANT_NAMES:
            if variant in [VARIANT_NAMES.none, VARIANT_NAMES.ancestral]:
                sampled_params[f'kappa_{variant}_{measure}'] = 1.0
            elif variant in [VARIANT_NAMES.omicron, VARIANT_NAMES.ba5, VARIANT_NAMES.omega]:
                sampled_params[f'kappa_{variant}_{measure}'] = variant_rr[f'{variant}_{rate}']
            else:
                sampled_params[f'kappa_{variant}_{measure}'] = variant_rr[rate]
    return sampled_params, phi_matrix


def sample_idr_parameters(rates_parameters: RatesParameters, draw_id: int) -> Dict[str, float]:
    rates_parameters = rates_parameters.to_dict()
    params = {}
    param_names = [
        'p_asymptomatic_pre_omicron',
        'p_asymptomatic_post_omicron',
        'minimum_asymptomatic_idr_fraction',
        'maximum_asymptomatic_idr',
    ]
    for parameter in param_names:
        param_spec = rates_parameters[parameter]
        if isinstance(param_spec, (int, float)):
            value = param_spec
        else:
            value = sample_parameter(parameter, draw_id, *param_spec)
        params[parameter] = value

    return params


def sample_antiviral_effectiveness(rates_parameters: RatesParameters, measure: str, draw_id: int) -> float:
    rates_parameters = rates_parameters.to_dict()
    if measure == 'case':
        parameter = 'antiviral_effectiveness_idr'
    elif measure == 'admission':
        parameter = 'antiviral_effectiveness_ihr'
    elif measure == 'death':
        parameter = 'antiviral_effectiveness_ifr'

    param_spec = rates_parameters[parameter]
    if isinstance(param_spec, (int, float)):
        value = param_spec
    else:
        value = sample_parameter(parameter, draw_id, *param_spec)

    return value


def sample_parameter(parameter: str, draw_id: int, lower: float, upper: float) -> float:
    random_state = utilities.get_random_state(f'{parameter}_{draw_id}')
    return random_state.uniform(lower, upper)

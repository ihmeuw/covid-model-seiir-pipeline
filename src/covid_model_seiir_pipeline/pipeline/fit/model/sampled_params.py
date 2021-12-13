from typing import NamedTuple, Dict

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import utilities
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.pipeline.fit.specification import (
    RatesParameters,
    FitParameters,
)


class Durations(NamedTuple):
    exposure_to_case: int
    exposure_to_admission: int
    exposure_to_seroconversion: int
    exposure_to_death: int
    pcr_to_seropositive: int
    admission_to_seropositive: int
    seropositive_to_death: int
    max_lag: int


def sample_durations(params: RatesParameters, draw_id: int) -> Durations:
    random_state = utilities.get_random_state(f'epi_durations_draw_{draw_id}')

    exposure_to_admission = random_state.choice(list(params.exposure_to_admission))
    exposure_to_seroconversion = random_state.choice(list(params.exposure_to_seroconversion))
    admission_to_death = random_state.choice(list(params.admission_to_death))
    max_lag = max(list(params.exposure_to_admission)) + max(list(params.admission_to_death))

    return Durations(
        exposure_to_case=exposure_to_admission,
        exposure_to_admission=exposure_to_admission,
        exposure_to_seroconversion=exposure_to_seroconversion,
        exposure_to_death=exposure_to_admission + admission_to_death,
        pcr_to_seropositive=exposure_to_seroconversion - exposure_to_admission,
        admission_to_seropositive=exposure_to_seroconversion - exposure_to_admission,
        seropositive_to_death=(exposure_to_admission + admission_to_death) - exposure_to_seroconversion,
        max_lag=max_lag,
    )


class VariantRR(NamedTuple):
    ifr: float
    ihr: float
    idr: float


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
    return VariantRR(**rrs)


def sample_day_inflection(params: RatesParameters, draw_id: int) -> pd.Timestamp:
    random_state = utilities.get_random_state(f'day_inflection_draw_{draw_id}')
    return pd.Timestamp(str(random_state.choice(params.day_inflection_options)))


def sample_ode_params(variant_rr: VariantRR, beta_fit_params: FitParameters, draw_id: int) -> Dict[str, float]:
    beta_fit_params = beta_fit_params.to_dict()

    sampled_params = {}
    for parameter in beta_fit_params:
        param_spec = beta_fit_params[parameter]
        if isinstance(param_spec, (int, float)):
            value = param_spec
        else:
            value = sample_parameter(parameter, draw_id, *param_spec)

        key = parameter if 'kappa' in parameter else f'{parameter}_all_infection'
        sampled_params[key] = value

    for measure, rate in (('death', 'ifr'), ('admission', 'ihr'), ('case', 'idr')):
        rr = variant_rr._asdict()[rate]
        for variant in VARIANT_NAMES:
            if variant in [VARIANT_NAMES.none, VARIANT_NAMES.ancestral]:
                sampled_params[f'kappa_{variant}_{measure}'] = 1.0
            elif variant in VARIANT_NAMES.omicron:
                sampled_params[f'kappa_{variant}_{measure}'] = rr * 0.2
            else:
                sampled_params[f'kappa_{variant}_{measure}'] = rr
    return sampled_params


def sample_parameter(parameter: str, draw_id: int, lower: float, upper: float) -> float:
    random_state = utilities.get_random_state(f'{parameter}_{draw_id}')
    return random_state.uniform(lower, upper)

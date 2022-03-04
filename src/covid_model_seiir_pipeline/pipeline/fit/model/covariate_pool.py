import itertools
from typing import Dict, List
import yaml

from loguru import logger

from covid_model_seiir_pipeline.lib import utilities

# Get from config?
COVARIATE_POOL = (
    'obesity',
    'smoking_prevalence',
    'diabetes',
    'ckd',
    'cancer',
    'copd',
    'cvd',
    'uhc',
    'haqi',
    'prop_65plus'
)


def make_covariate_pool(n_samples: int) -> Dict[str, Dict[int, List[str]]]:
    with open('/ihme/covid-19/rates-covariates/2021_12_19.01/covariate_combinations.yaml', 'r') as file:
        selected_combinations = yaml.full_load(file)
    if not all([c in COVARIATE_POOL for sc in selected_combinations for c in sc]):
        raise ValueError('Invalid covariate selected.')
    random_state = utilities.get_random_state('ihr_and_ifr_covariate_pool')
    selected_combinations = random_state.choice(selected_combinations, n_samples)

    idr_covariate_options = [['haqi'], ['uhc'], ['prop_65plus'], [], ]
    random_state = utilities.get_random_state('idr_covariate_pool')
    idr_covariate_pool = random_state.choice(idr_covariate_options, n_samples)

    covariate_selections = {'ifr': {}, 'ihr': {}, 'idr': {}}
    for draw in range(n_samples):
        covariate_selections['ifr'][draw] = selected_combinations[draw]
        covariate_selections['ihr'][draw] = selected_combinations[draw]
        covariate_selections['idr'][draw] = idr_covariate_pool[draw]

    return covariate_selections

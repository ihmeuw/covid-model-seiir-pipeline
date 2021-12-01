import itertools
from typing import Dict, List

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
    test_combinations = []
    for i in range(len(COVARIATE_POOL)):
        test_combinations += [list(set(cc)) for cc in itertools.combinations(COVARIATE_POOL[:-1], i + 1)]
    test_combinations = [cc for cc in test_combinations if
                         len([c for c in cc if c in ['uhc', 'haqi']]) <= 1]
    logger.warning('Not actually testing covariate combinations.')
    selected_combinations = [tc for tc in test_combinations if 'smoking_prevalence' in tc and len(tc) >= 5][:n_samples]

    idr_covariate_options = [['haqi'], ['uhc'], ['prop_65plus'], [], ]
    random_state = utilities.get_random_state('idr_covariate_pool')
    idr_covariate_pool = random_state.choice(idr_covariate_options, n_samples)

    covariate_selections = {'ifr': {}, 'ihr': {}, 'idr': {}}
    for draw in range(n_samples):
        covariate_selections['ifr'][draw] = selected_combinations[draw]
        covariate_selections['ihr'][draw] = selected_combinations[draw]
        covariate_selections['idr'][draw] = idr_covariate_pool[draw]

    return covariate_selections

"""ETL for testing covariates and scenarios."""
from pathlib import Path
from typing import Dict

import pandas as pd

from covid_shared import paths as paths

from covid_input_seir_covariates.utilities import CovariateGroup, check_schema


COVARIATE_NAMES = (
    'testing',
)

DEFAULT_OUTPUT_ROOT = paths.TESTING_OUTPUT_ROOT


def get_covariates(covariates_root: Path) -> CovariateGroup:
    """Gathers and formats all testing covariates and scenarios."""
    data_path = covariates_root / 'forecast_raked_test_pc_simple.csv'
    scenario_map_path = covariates_root / 'testing_scenario_dict.csv'
    testing_data = load_testing_data(data_path)
    mapping = load_scenario_mapping_data(scenario_map_path)

    scenarios = {}
    for k, v in mapping.items():
        scenarios[v] = testing_data[k].rename(f'testing_{v}').reset_index()
    info = {}

    return {'testing': (scenarios, info)}


def load_testing_data(path: Path) -> pd.DataFrame:
    """Loads, validates, and cleans testing data.

    The raw testing data file contains all scenarios.

    """
    data = pd.read_csv(path)

    acceptable_column_sets = [
        {'location_id', 'location_name', 'observed', 'date',
         'test_pc', 'test_pc_better', 'test_pc_worse'},
        # Schema change on 5/15
        {'location_id', 'location_name', 'observed', 'date',
         'test_pc', 'test_pc_better', 'test_pc_worse', 'pop'},
        # Schema change on 8/20
        {'location_id', 'location_name', 'observed', 'date',
         'test_pc', 'test_pc_better', 'test_pc_worse', 'pop', 'frontier'},
        # Schema change on 12/8
        {'location_id', 'location_name', 'observed', 'date',
         'test_pc', 'test_pc_better', 'test_pc_worse', 'pop', 'frontier',
         'population'},
    ]
#    check_schema(data, acceptable_column_sets, path)

    data['date'] = pd.to_datetime(data['date'])
    data['observed'] = data['observed'].astype(int)
    data = data.set_index(['location_id', 'date', 'observed']).sort_index()
    return data


def load_scenario_mapping_data(path: Path) -> Dict[str, str]:
    """Loads the mapping between scenario name and testing data column."""
    mapping = pd.read_csv(path)
    mapping = mapping.set_index('column_name').scenario_name.to_dict()
    return mapping

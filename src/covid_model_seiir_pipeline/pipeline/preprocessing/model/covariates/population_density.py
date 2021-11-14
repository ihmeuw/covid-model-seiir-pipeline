from pathlib import Path

import pandas as pd

from covid_shared import paths as paths

from covid_input_seir_covariates.utilities import CovariateGroup, check_schema


_UNDER_1K_COLUMNS = ['<150 ppl/sqkm', '150-300 ppl/sqkm', '300-500 ppl/sqkm', '500-1000 ppl/sqkm']
_UNDER_2_5K_COLUMNS = _UNDER_1K_COLUMNS + ['1000-2500 ppl/sqkm']
_UNDER_5K_COLUMNS = _UNDER_2_5K_COLUMNS + ['2500-5000 ppl/sqkm']


_POPULATION_DENSITY_COVARIATE_MAP = {
    'proportion_over_1k': _UNDER_1K_COLUMNS,
    'proportion_over_2_5k': _UNDER_2_5K_COLUMNS,
    'proportion_over_5k': _UNDER_5K_COLUMNS,
}

COVARIATE_NAMES = tuple(_POPULATION_DENSITY_COVARIATE_MAP.keys())

DEFAULT_OUTPUT_ROOT = paths.POPULATION_DENSITY_OUTPUT_ROOT


def get_covariates(covariates_root: Path) -> CovariateGroup:
    pop_density_path = covariates_root / 'all_outputs_2020_full.csv'
    pop_density = load_population_density_data(pop_density_path)

    population_density_covariates = {}
    for covariate_name, exclude_columns in _POPULATION_DENSITY_COVARIATE_MAP.items():
        covariate = (pop_density
                     .drop(columns=exclude_columns)
                     .sum(axis=1)
                     .reset_index()
                     .rename(columns={0: f'{covariate_name}_reference'}))
        covariate['observed'] = 1
        scenarios = {
            'reference': covariate
        }
        info = {}
        population_density_covariates[covariate_name] = (scenarios, info)

    return population_density_covariates


def load_population_density_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)

    acceptable_column_sets = [
        {'Unnamed: 0', 'location_id', 'ihme_loc_id',
         'location_name', 'year_id', 'pop_density', 'pop_proportion'}
    ]
    check_schema(data, acceptable_column_sets, path)

    data = data.set_index(['location_id', 'pop_density']).pop_proportion.unstack()
    return data

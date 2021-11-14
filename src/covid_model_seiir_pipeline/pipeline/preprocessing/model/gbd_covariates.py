from pathlib import Path

import pandas as pd

from covid_shared import paths as paths

from covid_input_seir_covariates.utilities import CovariateGroup, check_schema


COVARIATE_NAMES = (
    'air_pollution_pm_2_5',
    'lri_mortality',
    'proportion_under_100m',
    'smoking_prevalence',
)

DEFAULT_OUTPUT_ROOT = paths.MODEL_INPUTS_ROOT


def get_covariates(covariates_root: Path) -> CovariateGroup:
    cov_root = covariates_root / 'gbd_covariates'
    gbd_covariates = {}
    for covariate_name in COVARIATE_NAMES:
        covariate_path = cov_root / f'{covariate_name}.csv'
        covariate = _load_gbd_covariate_data(covariate_path)
        scenarios = {
            'reference': covariate
        }
        info = {}
        gbd_covariates[covariate_name] = (scenarios, info)
    return gbd_covariates


def _load_gbd_covariate_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    covariate_name = path.stem
    acceptable_column_sets = [
        {'location_id', 'observed', covariate_name},
    ]
    check_schema(data, acceptable_column_sets, path)
    data = data.rename(columns={covariate_name: f'{covariate_name}_reference'})
    return data

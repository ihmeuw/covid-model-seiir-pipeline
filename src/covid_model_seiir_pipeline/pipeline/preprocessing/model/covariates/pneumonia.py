from pathlib import Path

import pandas as pd

from covid_shared import paths as paths

from covid_input_seir_covariates.utilities import CovariateGroup, check_schema


COVARIATE_NAMES = (
    'pneumonia',
)

DEFAULT_OUTPUT_ROOT = paths.PNEUMONIA_OUTPUT_ROOT


def get_covariates(covariates_root: Path) -> CovariateGroup:
    pneumonia_path = covariates_root / 'pneumonia.csv'
    pneumonia_data = _load_pneumonia_data(pneumonia_path)
    scenarios = {
        'reference': pneumonia_data
    }
    info = {}
    return {'pneumonia': (scenarios, info)}


def _load_pneumonia_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    acceptable_column_sets = [
        set(['date', 'location_id', 'observed', 'value']
            + [f'draw_{i}' for i in range(1000)])
    ]
    check_schema(data, acceptable_column_sets, path)
    data['date'] = pd.to_datetime(data['date'])
    data['observed'] = float('nan')
    data = (data
            .loc[:, ['date', 'location_id', 'observed', 'value']]
            .rename(columns={'value': 'pneumonia_reference'}))
    next_year = data.copy()
    next_year['date'] += pd.Timedelta(days=366)
    next_year = next_year.groupby("location_id", as_index=False).apply(lambda x : x.iloc[1:-1]).reset_index(drop=True)
    year_after_next = next_year.copy()
    year_after_next['date'] += pd.Timedelta(days=365)
    data = pd.concat([data, next_year, year_after_next]).sort_values(["location_id", "date"]).reset_index(drop=True)
    return data

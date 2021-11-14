from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

from covid_shared import paths as paths

from covid_input_seir_covariates.utilities import CovariateGroup, check_schema


COVARIATE_NAMES = (
    'mobility',
)

DEFAULT_OUTPUT_ROOT = paths.MOBILITY_COVARIATES_OUTPUT_ROOT


def get_covariates(covariates_root: Path) -> CovariateGroup:
    mobility_reference_path = covariates_root / 'mobility_reference.csv'
    mobility_worse_path = covariates_root / 'mobility_vaccine_adjusted.csv'
    mobility_effect_path = covariates_root / 'mobility_mandate_coefficients.csv'

    metadata_path = covariates_root / 'metadata.yaml'
    with metadata_path.open() as metadata_file:
        metadata = yaml.full_load(metadata_file)
        reference_sd_lift_path = Path(metadata['sd_lift_path']) / 'percent_mandates_reference.csv'
        worse_sd_lift_path = Path(metadata['sd_lift_path']) / 'percent_mandates_vaccine_adjusted.csv'

    reference_mobility_data = load_mobility_data(mobility_reference_path)
    worse_mobility_data = load_mobility_data(mobility_worse_path)    

    mobility_effect = load_mobility_effect_data(mobility_effect_path)

    reference_sd_lift = load_sd_lift_data(reference_sd_lift_path)
    worse_sd_lift = load_sd_lift_data(worse_sd_lift_path)    

    mobility_reference = reference_mobility_data.loc[:, ['location_id', 'date', 'observed', 'mobility_reference']]
    mobility_worse = worse_mobility_data.loc[:, ['location_id', 'date', 'observed', 'mobility_reference']].rename(columns={'mobility_reference': 'mobility_worse'})

    scenarios = {
        'reference': mobility_reference,
        'worse': mobility_worse,
    }
    info = {
        'reference_mandate_lift': reference_sd_lift,
        'worse_mandate_lift': worse_sd_lift,
        'effect': mobility_effect,
    }
    return {'mobility': (scenarios, info)}


def split_mobility(mobility_reference: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Get the minimum mobility value and the last date the value was at the
    # minimum by location.
    minimum_mobility = (mobility_reference
                        .loc[:, ['location_id', 'date', 'mobility_reference']]
                        .groupby('location_id')
                        .apply(lambda x: x.loc[x['mobility_reference'] == x['mobility_reference'].min()].iloc[-1])
                        .reset_index(drop=True)
                        .set_index('location_id'))
    mobility_prediction = mobility_reference.set_index(['location_id'])
    mobility_down, mobility_up = mobility_prediction.copy(), mobility_prediction.copy()
    down_mask = mobility_prediction.date > minimum_mobility.loc[mobility_prediction.index].date
    up_mask = mobility_prediction.date < minimum_mobility.loc[mobility_prediction.index].date

    mobility_down.loc[down_mask, 'mobility_reference'] = minimum_mobility['mobility_reference']
    mobility_up.loc[up_mask, 'mobility_reference'] = minimum_mobility['mobility_reference']
    mobility_down = mobility_down.rename(columns={'mobility_reference': 'mobility_down'})
    mobility_up = mobility_up.rename(columns={'mobility_reference': 'mobility_up'})
    return mobility_down.reset_index(), mobility_up.reset_index()


def load_mobility_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data['observed'] = (1 - data['type']).astype(int)
    data['location_id'] = data['location_id'].astype(int)
    data['date'] = pd.to_datetime(data['date'])

    output_columns = ['location_id', 'date', 'observed', 'mobility_reference']
    data = (data.rename(columns={'mobility_forecast': 'mobility_reference'})
            .loc[:, output_columns]
            .sort_values(['location_id', 'date']))

    return data


def load_mobility_effect_data(path: Path) -> pd.DataFrame:
    effect_cols = ['sd1', 'sd2', 'sd3', 'psd1', 'psd3', 'anticipate']
    output_columns = ['location_id'] + effect_cols

    data = pd.read_csv(path)

    data['location_id'] = data['location_id'].astype(int)
    data = (data.rename(columns={f'{effect}_eff': effect for effect in effect_cols})
            .loc[:, output_columns]
            .sort_values(['location_id']))
    return data


def load_sd_lift_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)

    output_columns = ['location_id', 'date', 'percent']
    data = data[output_columns + ['percent_mandates']]
    data['date'] = pd.to_datetime(data['date'])
    data['location_id'] = data['location_id'].astype(int)

    # adjust the mandate lift so s hemisphere locs don't have duplicated rows
    # Hack from Haley:
    # percent_mandates has a value when observed, so use that to subset out duplicated
    s_hemisphere_locs = data.loc[data['percent_mandates'].notnull(), 'location_id'].unique()
    in_s_hemisphere = data['location_id'].isin(s_hemisphere_locs)
    first_observed = (data[in_s_hemisphere]
                      .groupby('location_id')['date']
                      .min())
    first_predicted = (data[in_s_hemisphere & data['percent_mandates'].isnull()]
                       .groupby('location_id')['date']
                       .min())
    date_shift = (first_predicted - first_observed).rename('shift').reset_index()

    # Subset to just the projected time series
    data = data[data['percent_mandates'].isnull()]
    data = data.merge(date_shift, how='outer', left_on='location_id', right_on='location_id')
    data.loc[in_s_hemisphere, 'date'] = data.loc[in_s_hemisphere, 'date'] - data.loc[in_s_hemisphere, 'shift']

    return data[output_columns]

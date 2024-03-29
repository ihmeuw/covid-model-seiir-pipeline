from typing import List, Tuple
from loguru import logger

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.side_analysis.npi_location_splitting.data import (
    DataLoader,
)

SMOOTHING_DAYS = 11
MEASURES = ['cases', 'deaths']


def get_prod_location_label(npi_hierarchy: pd.DataFrame, prod_hierarchy: pd.DataFrame) -> pd.DataFrame:
    prod_locations = prod_hierarchy['location_id'].astype(str).to_list()
    npi_hierarchy['prod_location_id'] = npi_hierarchy['path_to_top_parent'].apply(
        lambda x: int([xi for xi in x.split(',') if xi in prod_locations][-1])
    )
    npi_hierarchy.loc[npi_hierarchy['level'] <= 3, 'prod_location_id'] = npi_hierarchy['location_id']

    return npi_hierarchy.set_index('location_id').loc[:, ['prod_location_id']]


def fill_time_series(loc_data: pd.DataFrame, dates: pd.DataFrame):
    start_date, end_date = dates.loc[loc_data['prod_location_id'].unique().item()]
    start_date -= pd.Timedelta(days=1)  # plop a 0 in front to start interpolation
    loc_data = (loc_data
                .reset_index('location_id', drop=True)
                .drop('prod_location_id', axis=1))
    loc_data = loc_data.reindex(pd.Index(pd.date_range(start_date, end_date), name='date'))
    loc_data.iloc[0] = 0
    loc_data = loc_data.interpolate().ffill()
    loc_data = loc_data.drop(start_date)

    return loc_data


def process_measure_data(
    measure: str,
    data_loader: DataLoader,
    npi_hierarchy: pd.DataFrame,
    prod_hierarchy: pd.DataFrame,
    prod_infections: pd.Series,
    smooth: bool = True,
    which_locs: str = 'all',
) -> pd.Series:
    # load data, convert to daily and drop missingness and days <= 0, revert to cumulative
    raw_data = data_loader.load_raw_data(which_locs, measure, npi_hierarchy)
    data = raw_data.dropna()
    data = data.groupby('location_id').diff().fillna(data)
    data = data.loc[data > 0]
    data = data.groupby('location_id').cumsum()

    # reindex to date of exposure
    time_shift = {
        'cases': 12,
        'deaths': 26,
    }
    data = data.reset_index('date')
    data['date'] -= pd.Timedelta(days=time_shift[measure])
    data = data.set_index('date', append=True).sort_index()

    # attach state label to counties
    data = get_prod_location_label(npi_hierarchy, prod_hierarchy).join(data, how='right')

    # get start and end dates by state
    start_dates = (prod_infections
                   .loc[prod_infections.groupby('prod_location_id').cumsum() > 0].reset_index('date')
                   .groupby('prod_location_id').first()['date']
                   .rename('start_date'))
    end_dates = (data
                 .reset_index('date')
                 .groupby('prod_location_id').last()['date']
                 .rename('end_date'))
    dates = pd.concat([start_dates, end_dates], axis=1)

    # interpolate missingness and 0s, enforce common start and end date within a state
    data = data.groupby('location_id').apply(fill_time_series, dates)

    # convert to daily
    data = data.groupby('location_id').diff().fillna(data).loc[:, measure]

    # get rolling average
    if smooth:
        data = (data
                .groupby('location_id').apply(lambda x: x.rolling(center=True,
                                                                  window=SMOOTHING_DAYS,
                                                                  min_periods=SMOOTHING_DAYS).mean()))

    return raw_data, data


def add_brazil_residual(
    measure: str,
    data_loader: DataLoader,
    npi_hierarchy: pd.DataFrame,
    prod_hierarchy: pd.DataFrame,
    prod_infections: pd.Series,
    processed_data: pd.Series,
) -> Tuple[pd.Series]:
    _, admin1_data = process_measure_data(measure,
                                          data_loader,
                                          npi_hierarchy,
                                          prod_hierarchy,
                                          prod_infections,
                                          which_locs='bra_admin1')

    _, admin2_data = process_measure_data(measure,
                                          data_loader,
                                          npi_hierarchy,
                                          prod_hierarchy,
                                          prod_infections,
                                          which_locs='bra_admin2')

    agg_admin1_data = (get_prod_location_label(npi_hierarchy, prod_hierarchy)
                       .join(admin2_data, how='right')
                       .groupby(['prod_location_id', 'date'])[measure].sum())
    agg_admin1_data.index.names = ['location_id', 'date']

    idx = agg_admin1_data.index.union(admin1_data.index)

    residual_admin2_data = admin1_data.reindex(idx) - agg_admin1_data.reindex(idx)
    residual_admin2_data = residual_admin2_data.clip(0, np.inf)

    residual_admin2_data = residual_admin2_data.reset_index()
    residual_admin2_data['location_id'] = -residual_admin2_data['location_id']
    residual_admin2_data = residual_admin2_data.set_index(['location_id', 'date']).loc[:, measure]
    processed_data = processed_data.append(residual_admin2_data)

    is_bra = npi_hierarchy['path_to_top_parent'].apply(lambda x: '135' in x.split(','))
    is_admin1 = npi_hierarchy['level'] == 4
    residual_hierarchy = npi_hierarchy.loc[is_bra & is_admin1]
    residual_hierarchy['parent_id'] = residual_hierarchy['location_id']
    residual_hierarchy['location_id'] = -residual_hierarchy['location_id']
    residual_hierarchy['location_name'] += ' residual'
    residual_hierarchy['path_to_top_parent'] += ',' + residual_hierarchy['location_id'].astype(str)
    residual_hierarchy['level'] += 1
    residual_hierarchy['sort_order'] += 0.1
    npi_hierarchy = npi_hierarchy.append(residual_hierarchy).sort_values('sort_order').reset_index(drop=True)

    return npi_hierarchy, processed_data


def split_prod_infections(
    npi_hierarchy: pd.DataFrame,
    prod_hierarchy: pd.DataFrame,
    measure_data: pd.Series,
    prod_infections: pd.Series,
) -> pd.Series:
    # extend head/tail in order to get proportions for full time series
    measure_data = (measure_data
                    .rename('measure_data')
                    .groupby('location_id').ffill()
                    .groupby('location_id').bfill())

    # calculate proportions
    split_proportions = get_prod_location_label(npi_hierarchy, prod_hierarchy).join(measure_data, how='right')
    if split_proportions['prod_location_id'].isnull().any():
        raise ValueError('Some locations in data are not in hierarchy.')
    split_proportions = (split_proportions
                         .set_index('prod_location_id', append=True)
                         .reorder_levels(['prod_location_id', 'location_id', 'date'])
                         .loc[:, 'measure_data'])
    split_proportions /= split_proportions.groupby(['prod_location_id', 'date']).transform(sum)

    data_idx = split_proportions.reset_index('location_id', drop=True).index.drop_duplicates()

    npi_infections = split_proportions * prod_infections.loc[data_idx]
    npi_infections = (npi_infections
                      .reset_index('prod_location_id', drop=True)
                      .reorder_levels(['location_id', 'date'])
                      .sort_index()
                      .rename('daily_infections'))

    return npi_infections


def extrapolate_infections(
    npi_infections: pd.DataFrame,
    fill_to: str,
    fill_from: str
) -> pd.DataFrame:
    # how many transition days
    t_d = 30

    # get initial ratio
    ratio = (
        npi_infections[f'daily_infections_{fill_to}']
        / npi_infections[f'daily_infections_{fill_from}']
    ).rename('ratio')
    ratio_notna = ratio.dropna()
    counter = ratio_notna.notnull()
    # first_ratio = ratio_notna.groupby('location_id').first()
    last_ratio = ratio_notna.groupby('location_id').last()

    # # weight backcast ratio inverse chronologically (linear)
    # b_weights = counter.sort_index(ascending=False).groupby('location_id').cumsum()
    # b_weights /= b_weights.groupby('location_id').transform(sum)
    # b_ratio = (ratio_notna * b_weights.sort_index()).groupby('location_id').sum().replace(0, np.nan)

    # # create backcast values
    # b_idx = ~(ratio.sort_index().notnull().groupby('location_id').cummax())
    # b_fill = b_idx.sort_index(ascending=False).groupby('location_id').cumsum().sort_index().clip(0, t_d) / t_d
    # b_fill = b_fill.loc[b_fill > 0]
    # b_fill = b_fill.loc[b_fill == b_fill.groupby('location_id').transform(max)]
    # b_fill = first_ratio + (b_ratio - first_ratio) * b_fill

    # weight forecast ratio chronologically (linear)
    f_weights = counter.sort_index().groupby('location_id').cumsum()
    f_weights /= f_weights.groupby('location_id').transform(sum)
    f_ratio = (ratio_notna * f_weights.sort_index()).groupby('location_id').sum().replace(0, np.nan)

    # create forecast values
    f_idx = ~(ratio.sort_index(ascending=False).notnull().groupby('location_id').cummax().sort_index())
    f_fill = f_idx.groupby('location_id').cumsum().clip(0, t_d) / t_d
    f_fill = f_fill.loc[f_fill > 0]
    f_fill = f_fill.loc[f_fill == f_fill.groupby('location_id').transform(max)]
    f_fill = last_ratio + (f_ratio - last_ratio) * f_fill

    # combine, splice into original ratios, and interpolate
    #fill = pd.concat([b_fill, f_fill.drop(b_fill.index, errors='ignore')])
    fill = f_fill.copy()
    ratio = ratio.fillna(fill).groupby('location_id').apply(pd.Series.interpolate)

    # fill in original data
    npi_infections[f'daily_infections_{fill_to}'] = npi_infections[f'daily_infections_{fill_to}'].fillna(
        npi_infections[f'daily_infections_{fill_from}'] * ratio
    )

    return npi_infections


def generate_measure_specific_infections(
    data_loader: DataLoader,
    npi_hierarchy: pd.DataFrame,
    prod_hierarchy: pd.DataFrame,
    prod_infections: pd.Series,
    measures: List[str] = MEASURES,
) -> Tuple[pd.Series]:
    # load NPI hierarchy level outputs by measure
    raw_data, processed_data, npi_infections = [], [], []
    for measure in measures:
        logger.info(measure)
        # process input data
        _raw_data, _processed_data = process_measure_data(
            measure,
            data_loader,
            npi_hierarchy,
            prod_hierarchy,
            prod_infections
        )
        _npi_hierarchy, _processed_data = add_brazil_residual(
            measure,
            data_loader,
            npi_hierarchy,
            prod_hierarchy,
            prod_infections,
            _processed_data
        )

        # split up infections
        _infections = split_prod_infections(
            _npi_hierarchy,
            prod_hierarchy,
            _processed_data,
            prod_infections,
        ).rename(f'daily_infections_{measure}')

        raw_data.append(_raw_data)
        processed_data.append(_processed_data)
        npi_infections.append(_infections)
    raw_data = pd.concat(raw_data, axis=1)
    processed_data = pd.concat(processed_data, axis=1)
    npi_infections = pd.concat(npi_infections, axis=1)
    
    return raw_data, processed_data, npi_infections


def generate_measure_weights(processed_data: pd.DataFrame) -> pd.DataFrame:
    weights = (processed_data.dropna(how='all') > (1 / SMOOTHING_DAYS)).groupby('location_id').mean()
    weights = weights.divide(weights.sum(axis=1), axis=0)
    weights = weights.rename(columns={col: f'daily_infections_{col}' for col in weights})

    return weights


def combine_measure_specific_infections(npi_infections: pd.DataFrame,
                                        processed_data: pd.DataFrame) -> Tuple[pd.DataFrame]:
    # downweight based on sparseness
    weights = generate_measure_weights(processed_data)
    
    # fill out tails
    if any([measure not in ['cases', 'deaths'] for measure in MEASURES]):
        raise ValueError('Unexpected measure, revise extrapolation logic.')
    npi_infections_fill = npi_infections.copy()
    npi_infections_fill = extrapolate_infections(
        # fill in deaths first...
        npi_infections_fill,
        fill_to='deaths',
        fill_from='cases',
    )
    npi_infections_fill = extrapolate_infections(
        # ... then cases (if necessary, probably is not in most locations)
        npi_infections_fill,
        fill_to='cases',
        fill_from='deaths',
    )

    # combine
    npi_infections = (npi_infections
                      .join(npi_infections_fill.mean(axis=1).rename('daily_infections_composite'))
                      .join((npi_infections_fill * weights).sum(axis=1).rename('daily_infections_composite_weighted')))

    return weights, npi_infections

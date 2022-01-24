"""Postprocessing aggregation strategies."""
from typing import List, Optional

from loguru import logger
import numpy as np
import pandas as pd


def summarize(data: pd.DataFrame, metric: str = 'mean', ci: float = 0.95, ci2: float = None) -> pd.DataFrame:
    assert metric in ['mean', 'median']
    assert 0. < ci <= 1.
    if ci2 is not None:
        assert 0. < ci2 <= 1.

    upper = 1 - (1 - ci) / 2
    summary_metrics = [
        data.mean(axis=1).rename(metric),
        data.quantile(upper, axis=1).rename('upper'),
        data.quantile(1-upper, axis=1).rename('lower'),
    ]

    if ci2 is not None:
        upper = 1 - (1 - ci2) / 2
        summary_metrics.extend([
            data.quantile(upper, axis=1).rename('upper2'),
            data.quantile(1-upper, axis=1).rename('lower2'),
        ])

    return pd.concat(summary_metrics, axis=1)


def fill_cumulative_date_index(data: pd.DataFrame, date_index: Optional[pd.Index]) -> pd.DataFrame:
    """Fills missing dates in cumulative data so in can be aggregated correctly."""
    if date_index is None:
        return data

    def _reindex(df):
        df = (df
              .reset_index(level='location_id', drop=True)
              .reindex(date_index)
              .fillna(method='ffill')
              .fillna(method='bfill'))
        return df

    return data.groupby(level='location_id').apply(_reindex)


def sum_aggregator(measure_data: pd.DataFrame,
                   hierarchy: pd.DataFrame,
                   _population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates most-detailed locations to locations in the hierarchy by sum.

    The ``_population`` argument is here for api consistency and is not used.

    """
    if 'observed' in measure_data.index.names:
        measure_data = measure_data.reset_index(level='observed')

    agg_levels = []
    date_index = None
    if 'date' in measure_data.index.names:
        agg_levels.append('date')

        dates = measure_data.reset_index().set_index('location_id').date
        min_date, max_date = dates.min(), dates.max()
        lower_missing = np.any(dates.groupby('location_id').min() > min_date)
        upper_missing = np.any(dates.groupby('location_id').max() < max_date)
        if lower_missing or upper_missing:
            date_index = pd.Index(pd.date_range(min_date, max_date), name='date')

    for col in ['sex_id', 'year_id', 'round',
                'age_group_id', 'age_start', 'age_group_years_start', 'age_group_years_end']:
        if col in measure_data.index.names:
            agg_levels.append(col)

    levels = sorted(hierarchy['level'].unique().tolist())
    data_locs = _get_data_locs(measure_data, hierarchy)
    for level in reversed(levels[1:]):  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]

        for parent_id, children in locs_at_level.groupby('parent_id'):
            if parent_id in data_locs:
                continue
            child_locs = children.location_id.tolist()
            modeled_child_locs = list(set(child_locs).intersection(data_locs))
            if not modeled_child_locs:
                continue

            filled_measure_data = fill_cumulative_date_index(measure_data.loc[modeled_child_locs], date_index)
            aggregate = filled_measure_data.groupby(agg_levels).sum()
            aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])

            measure_data = measure_data.append(aggregate)
            data_locs.append(parent_id)

    # We'll call any aggregate with at least one observed point observed.
    if 'observed' in measure_data.columns:
        measure_data.loc[measure_data['observed'] > 0, 'observed'] = 1
        measure_data = measure_data.set_index('observed', append=True)
    return measure_data.sort_index()


def mean_aggregator(measure_data: pd.DataFrame,
                    hierarchy: pd.DataFrame,
                    population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates most-detailed locations to locations in the hierarchy by
    population-weighted mean.

    """
    if 'observed' in measure_data.index.names:
        measure_data = measure_data.reset_index(level='observed')
    # Get all age/sex population and append to the data.
    measure_columns = measure_data.columns
    measure_data = measure_data.join(population)

    data_locs = _get_data_locs(measure_data, hierarchy)
    # We'll work in the space of pop*measure where aggregation is just a sum.
    weighted_measure_data = measure_data.copy()
    weighted_measure_data[measure_columns] = measure_data[measure_columns].mul(measure_data['population'], axis=0)

    levels = sorted(hierarchy['level'].unique().tolist())
    for level in reversed(levels[1:]):  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]
        for parent_id, children in locs_at_level.groupby('parent_id'):
            if parent_id in data_locs:
                continue
            child_locs = children.location_id.tolist()
            modeled_child_locs = list(set(child_locs).intersection(data_locs))
            if not modeled_child_locs:
                continue

            if 'date' in weighted_measure_data.index.names:
                aggregate = weighted_measure_data.loc[modeled_child_locs].groupby(level='date').sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])
            else:
                aggregate = weighted_measure_data.loc[modeled_child_locs].sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id']).unstack()

            weighted_measure_data = weighted_measure_data.append(aggregate)
            aggregate = aggregate[measure_columns].div(aggregate['population'], axis=0)
            measure_data = measure_data.append(aggregate)

            data_locs.append(parent_id)

    measure_data = measure_data.drop(columns='population')
    # We'll call any aggregate with at least one observed point observed.
    if 'observed' in measure_data.columns:
        measure_data.loc[measure_data['observed'] > 0, 'observed'] = 1
        measure_data = measure_data.set_index('observed', append=True)
    return measure_data.sort_index()

def _get_data_locs(measure_data: pd.DataFrame, hierarchy: pd.DataFrame) -> List[int]:
    modeled_locs = set(measure_data.reset_index().location_id.unique().tolist())

    non_most_detailed_h = hierarchy.loc[hierarchy.most_detailed == 0, 'location_id'].tolist()
    non_most_detailed_m = list(set(modeled_locs).intersection(non_most_detailed_h))
    if non_most_detailed_m:
        logger.warning(f'Non most-detailed locations {sorted(non_most_detailed_m)} found in data. '
                       'These locations will not be aggregated.')
    data_locs = list(modeled_locs.intersection(hierarchy.location_id))
    return data_locs

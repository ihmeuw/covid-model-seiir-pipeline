from typing import List

from loguru import logger
import pandas as pd


def fill_dates(data: pd.DataFrame, interp_vars: List[str]) -> pd.DataFrame:
    data = data.set_index('date').sort_index()
    data = data.asfreq('D').reset_index()
    data[interp_vars] = data[interp_vars].interpolate(axis=0)
    data['location_id'] = (data['location_id']
                           .fillna(method='pad')
                           .astype(int))

    return data[['location_id', 'date'] + interp_vars]


def check_for_duplicate_dates(data: pd.DataFrame, name: str, groupby_cols: List[str] = ('location_id',)):
    groupby_cols = list(groupby_cols)
    data = data.reset_index()
    # Make sure we can do the check.
    assert not set(groupby_cols + ['date']).difference(data.columns)

    duplicates = data.groupby(groupby_cols).apply(lambda x: x[x.date.duplicated()])
    if not duplicates.empty:
        raise ValueError(f'Duplicates found in data set {name}: {duplicates}.')


def str_fmt(str_col: pd.Series) -> pd.Series:
    fmt_str_col = str_col.copy().str.lower().str.strip()
    fmt_str_col = fmt_str_col.str.lower()
    fmt_str_col = fmt_str_col.str.strip()

    return fmt_str_col


def validate_hierarchies(mr_hierarchy: pd.DataFrame, pred_hierarchy: pd.DataFrame) -> pd.DataFrame:
    mr = mr_hierarchy.loc[:, ['location_id', 'path_to_top_parent']]
    mr = mr.rename(columns={'path_to_top_parent': 'mr_path'})
    pred = pred_hierarchy.loc[:, ['location_id', 'path_to_top_parent']]
    pred = pred.rename(columns={'path_to_top_parent': 'pred_path'})

    data = mr.merge(pred, how='left')
    is_missing = data['pred_path'].isnull()
    is_different = (data['mr_path'] != data['pred_path']) & (~is_missing)

    if is_different.sum() > 0:
        raise ValueError(f'Some locations have a conflicting path to top parent in the mr and pred hierarchies:\n'
                         f'{data.loc[is_different]}.')

    if is_missing.sum() > 0:
        logger.warning(f'Some mr locations are missing in pred hierarchy and will be added:\n{data.loc[is_missing]}.')
        missing_locations = data.loc[is_missing, 'location_id'].to_list()
        missing_locations = mr_hierarchy.loc[mr_hierarchy['location_id'].isin(missing_locations)]
        pred_hierarchy = pred_hierarchy.append(missing_locations)
        most_detailed = ~pred_hierarchy.location_id.isin(pred_hierarchy.parent_id)
        pred_hierarchy.loc[:, 'most_detailed'] = 0
        pred_hierarchy.loc[most_detailed, 'most_detailed'] = 1

    return pred_hierarchy


def parent_inheritance(data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    if 'location_id' in data.index.names:
        index_names = data.index.names
        data = data.reset_index()
    else:
        assert 'location_id' in data.columns
        index_names = None

    location_ids = hierarchy['location_id'].to_list()
    path_to_top_parents = [list(reversed(p.split(',')[:-1]))
                           for p in hierarchy['path_to_top_parent'].to_list()]

    for location_id, path_to_top_parent in zip(location_ids, path_to_top_parents):
        if location_id not in data.reset_index()['location_id'].to_list():
            for parent_id in path_to_top_parent:
                try:
                    parent_data = data.set_index('location_id').loc[parent_id]
                    parent_data['location_id'] = location_id
                    data = data.append(parent_data)
                    break
                except KeyError:
                    pass
            else:
                location_name = hierarchy.set_index('location_id').location_name.loc[location_id]
                logger.warning(f'No data available for {location_name} or any of its parents.')

    if index_names is not None:
        data = data.set_index(index_names).sort_index()
    else:
        data = data.sort_values('location_id')
    return data

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


def aggregate_data_from_md(data: pd.DataFrame, hierarchy: pd.DataFrame, agg_var: str) -> pd.Series:
    if data[agg_var].max() <= 1:
        raise ValueError(f'Data in {agg_var} looks like rates - need counts for aggregation.')

    data = data.copy()

    is_md = hierarchy['most_detailed'] == 1
    md_location_ids = hierarchy.loc[is_md, 'location_id'].to_list()
    parent_location_ids = hierarchy.loc[~is_md, 'location_id'].to_list()

    md_data = data.loc[data['location_id'].isin(md_location_ids)]

    md_child_ids_lists = [(hierarchy
                           .loc[is_md & (
        hierarchy['path_to_top_parent'].apply(lambda x: str(parent_location_id) in x.split(','))),
                                'location_id']
                           .to_list()) for parent_location_id in parent_location_ids]
    parent_children_pairs = list(zip(parent_location_ids, md_child_ids_lists))

    parent_data = [aggregate(md_data.loc[md_data['location_id'].isin(md_child_ids)],
                             parent_id, md_child_ids, agg_var)
                   for parent_id, md_child_ids in parent_children_pairs]
    data = pd.concat([md_data] + parent_data)

    return data.reset_index(drop=True)


def aggregate(data: pd.DataFrame, parent_id: int, md_child_ids: List[int], agg_var: str) -> pd.Series:
    # not most efficient to go from md each time, but safest since dataset is not square (and not a ton of data)
    draw_level = 'draw' in data.columns
    if data.empty:
        if draw_level:
            return data.loc[:, ['draw', 'location_id', 'date', agg_var]]
        else:
            return data.loc[:, ['location_id', 'date', agg_var]]
    else:
        # REQUIRE ALL CHILD LOCATIONS
        if draw_level:
            data = data.groupby(['draw', 'date'])[agg_var].agg(['sum', 'count'])
            data['count'] = data.groupby('date')['count'].transform(min)
        else:
            data = data.groupby('date')[agg_var].agg(['sum', 'count'])
        is_complete = data['count'] == len(md_child_ids)
        data = data.loc[is_complete, 'sum'].rename(agg_var).reset_index()
        data['location_id'] = parent_id

        if draw_level:
            data = data.loc[:, ['draw', 'location_id', 'date', agg_var]]
        else:
            data = data.loc[:, ['location_id', 'date', agg_var]]

        return data


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

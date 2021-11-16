from typing import List

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

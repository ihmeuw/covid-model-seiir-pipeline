import itertools
from typing import Tuple

import numpy as np
import pandas as pd


def prepare_phis(past_infections: pd.Series, 
                 covariates: pd.DataFrame,
                 waning_matrix: pd.DataFrame,
                 waning_params: Tuple[float, int, float, int]):
    min_date = past_infections.reset_index().date.min()
    max_date = covariates.reset_index().date.max()
    
    dates = pd.Series(pd.date_range(min_date, max_date))
    times = (dates - min_date).dt.days    
    max_t = times.max() + 1
    
    phi_columns = [f'phi_{v_from}_{v_to}' 
                   for v_from, v_to in itertools.product(waning_matrix, waning_matrix)]
    phi = pd.DataFrame(0., 
                       index=times,
                       columns=phi_columns)
    
    w = make_raw_waning_dist(max_t, *waning_params)
    
    for v_from in waning_matrix.index:
        for v_to in waning_matrix.columns:
            phi.loc[:, f'phi_{v_from}_{v_to}'] = waning_matrix.at[v_from, v_to] * w
    
    return phi


def make_raw_waning_dist(max_t, l1, t1, l2, t2):
    w = np.zeros(max_t)
    w[:t1] = 1 + (l1 - 1) / (t1 - 0) * np.arange(t1)
    w[t1:t2] = l1 + (l2 - l1) / (t2 - t1) * np.arange(t2 - t1)
    w[t2:] = l2
    return w


def convert_date_index_to_time_index(data):
    data = data.sort_index()
    original_index = data.index
    idx_cols = original_index.names
    new_idx_cols = [c if c != 'date' else 't' for c in idx_cols]

    is_series = isinstance(data, pd.Series)

    data = data.reset_index()
    data['t'] = (data.date - data.date.min()).dt.days

    data = data.set_index(new_idx_cols).drop(columns='date')
    if is_series:
        data = data.iloc[:, 0]
    return data

import itertools

import pandas as pd


def prepare_phis(past_infections: pd.Series, 
                 covariates: pd.DataFrame,
                 waning_matrix: pd.DataFrame,
                 natural_waning_dist: pd.DataFrame):
    
    phi_columns = [f'phi_{v_from}_{v_to}' 
                   for v_from, v_to in itertools.product(waning_matrix, waning_matrix)]
    phi = pd.DataFrame(0.,
                       index=natural_waning_dist.index,
                       columns=phi_columns)
    
    for v_from in waning_matrix.index:
        for v_to in waning_matrix.columns:
            phi.loc[:, f'phi_{v_from}_{v_to}'] = waning_matrix.at[v_from, v_to] * w
    
    return phi




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

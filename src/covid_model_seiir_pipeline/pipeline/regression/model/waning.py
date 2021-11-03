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
            phi.loc[:, f'phi_{v_from}_{v_to}'] = waning_matrix.at[v_from, v_to] * natural_waning_dist
    return phi.loc['infection']


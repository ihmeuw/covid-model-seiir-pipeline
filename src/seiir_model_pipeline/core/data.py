import pandas as pd
from typing import List


def load_all_location_data(directories, location_ids, draw_id):
    dfs = dict()
    for loc in location_ids:
        file = directories.get_infection_file(location_id=loc, draw_id=draw_id)
        dfs[loc] = pd.read_csv(file)
    return dfs


def load_covariates(directories, covariate_names,
                    col_loc_id, col_date, col_observed,
                    location_id=None, forecasted=False):
    dfs = pd.DataFrame()
    for name in covariate_names:
        df = pd.read_csv(directories.get_covariate_file(name))
        if forecasted:
            df = df.loc[~df[col_observed]]
        else:
            df = df.loc[df[col_observed]]
        if dfs.empty:
            dfs = df
        else:
            dfs = dfs.merge(df, on=[col_loc_id, col_date])
    if location_id is not None:
        assert isinstance(location_id, List)
        dfs = dfs.loc[dfs[col_loc_id].isin(location_id)].copy()
    return dfs


def load_mr_coefficients(directories, draw_id, location_id):
    return pd.DataFrame()


def save_mr_coefficients(directories):
    pass


def load_peaked_dates(filepath, col_loc_id, col_date):
    df = pd.read_csv(filepath)
    return dict(zip(df[col_loc_id], df[col_date]))

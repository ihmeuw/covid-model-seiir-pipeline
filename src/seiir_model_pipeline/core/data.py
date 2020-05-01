import pandas as pd
import os
import numpy as np
from typing import List


def write_missing_infection_locations_file(directories, draw_id, location_ids):
    df = pd.DataFrame({
        'location_id': location_ids,
        'draw_id': draw_id
    })
    df.to_csv(directories.get_missing_infection_locations_file(draw_id))


def write_missing_covariate_locations_file(directories, covariate_dict):
    m = []
    for cov, locs in covariate_dict.items():
        m.append(
            pd.DataFrame({
                'covariate': cov,
                'location_ids': locs
            })
        )
    df = pd.concat(m).reset_index()
    df.to_csv(directories.get_missing_covariate_locations_file())


def load_all_location_data(directories, location_ids, draw_id):
    dfs = dict()
    missing_locations = []
    for loc in location_ids:
        file = directories.get_infection_file(location_id=loc, draw_id=draw_id)
        if not os.path.exists(file):
            missing_locations.append(loc)
        else:
            dfs[loc] = pd.read_csv(file)
    write_missing_infection_locations_file(directories, draw_id, missing_locations)
    return dfs


def format_covariates(directories, covariate_names,
                      col_loc_id, col_date, col_observed,
                      location_id=None):
    dfs = pd.DataFrame()
    missing_covariate_locations = {}
    for name in covariate_names:
        df = pd.read_csv(directories.get_covariate_file(name))
        cov_locations = df[col_loc_id].unique()
        if location_id is not None:
            missing_covariate_locations[name] = [x for x in location_id if x not in cov_locations]
        if dfs.empty:
            dfs = df
        else:
            # Need to check if col_observed is nan. If that's the case
            # then the covariate is static over time and we don't want to merge
            # on date or past/future, just the location
            if not np.isnan(df[col_observed]).all():
                dfs = dfs.merge(df, on=[col_loc_id, col_date, col_observed])
            else:
                dfs = dfs.merge(df, on=[col_loc_id])
    if location_id is not None:
        assert isinstance(location_id, List)
        dfs = dfs.loc[dfs[col_loc_id].isin(location_id)].copy()

    if location_id is not None:
        write_missing_covariate_locations_file(
            directories, covariate_dict=missing_covariate_locations
        )
    return dfs


def load_covariates(directories, col_loc_id, col_observed, location_id, forecasted=None):
    df = pd.read_csv(directories.get_cached_covariates_file())
    if forecasted is not None:
        df = df.loc[df[col_observed] == forecasted]
    if location_id is not None:
        assert isinstance(location_id, List)
        df = df.loc[df[col_loc_id].isin(location_id)].copy()
    return df


def cache_covariates(directories, covariate_names, col_loc_id, col_date, col_observed,
                     location_id):
    df = format_covariates(
        directories=directories, covariate_names=covariate_names, col_loc_id=col_loc_id,
        col_date=col_date, col_observed=col_observed, location_id=location_id
    )
    df.to_csv(directories.get_cached_covariates_file())


def load_mr_coefficients(directories, draw_id, location_id):
    return pd.DataFrame()


def save_mr_coefficients(directories):
    pass


def load_peaked_dates(filepath, col_loc_id, col_date):
    df = pd.read_csv(filepath)
    return dict(zip(df[col_loc_id], df[col_date]))

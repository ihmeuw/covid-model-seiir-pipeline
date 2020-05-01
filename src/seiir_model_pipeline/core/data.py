import pandas as pd
import numpy as np
from typing import List

from seiir_model_pipeline.core.versioner import FileDoesNotExist
from seiir_model_pipeline.core.versioner import COVARIATE_COL_DICT, INFECTION_COL_DICT


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
        try:
            file = directories.get_infection_file(location_id=loc, draw_id=draw_id)
            dfs[loc] = pd.read_csv(file)
        except FileDoesNotExist:
            missing_locations.append(loc)
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


def load_mr_coefficients(directories, draw_id):
    df = pd.read_csv(directories.get_draw_coefficient_file(draw_id))
    return df


def load_beta_fit(directories, draw_id, location_id):
    df = pd.read_csv(directories.get_draw_beta_fit_file(draw_id))
    df = df.loc[df[INFECTION_COL_DICT['COL_LOC_ID']] == location_id].copy()
    return df


def load_beta_params(directories, draw_id):
    df = pd.read_csv(directories.get_draw_beta_param_file(draw_id))
    return df.set_index('params')['values'].to_dict()


def load_peaked_dates(filepath, col_loc_id, col_date):
    df = pd.read_csv(filepath)
    return dict(zip(df[col_loc_id], df[col_date]))

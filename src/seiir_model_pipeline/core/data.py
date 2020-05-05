import pandas as pd
import yaml
import os
from typing import List

from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT


def get_missing_locations(directories, location_ids):
    infection_loc = [x.split('_')[-1] for x in os.listdir(directories.infection_dir)
                     if os.path.isdir(directories.infection_dir / x)]
    infection_loc = [int(x) for x in infection_loc if x.isdigit()]

    with open(directories.get_missing_covariate_locations_file()) as f:
        covariate_metadata = yaml.load(f, Loader=yaml.FullLoader)

    missing_covariate_loc = list()
    for k, v in covariate_metadata.items():
        missing_covariate_loc += v
    missing_covariate_loc = list(set(missing_covariate_loc))
    return [x for x in location_ids if x not in infection_loc or x in missing_covariate_loc]


def load_all_location_data(directories, location_ids, draw_id):
    dfs = dict()
    for loc in location_ids:
        file = directories.get_infection_file(location_id=loc, draw_id=draw_id)
        dfs[loc] = pd.read_csv(file)
    return dfs


def load_component_forecasts(directories, location_id, draw_id):
    df = pd.read_csv(
        directories.location_draw_component_forecast_file(
            location_id=location_id, draw_id=draw_id
        )
    )
    return df


def format_covariates(directories, covariate_names,
                      col_loc_id, col_date, col_observed,
                      location_id=None):
    dfs = pd.DataFrame()
    for name in covariate_names:
        df = pd.read_csv(directories.get_covariate_file(name))
        df = df.loc[~df[name].isnull()].copy()
        df.drop(columns=[col_observed], inplace=True, axis=1)
        if dfs.empty:
            dfs = df
        else:
            # time dependent covariates versus not
            if col_date in df.columns:
                dfs = dfs.merge(df, on=[col_loc_id, col_date])
            else:
                dfs = dfs.merge(df, on=[col_loc_id])
    if location_id is not None:
        assert isinstance(location_id, List)
        dfs = dfs.loc[dfs[col_loc_id].isin(location_id)].copy()
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
    df = pd.read_csv(directories.get_draw_beta_fit_file(location_id, draw_id))
    return df


def load_beta_params(directories, draw_id):
    df = pd.read_csv(directories.get_draw_beta_param_file(draw_id))
    return df.set_index('params')['values'].to_dict()


def load_peaked_dates(filepath, col_loc_id, col_date):
    df = pd.read_csv(filepath)
    return dict(zip(df[col_loc_id], df[col_date]))

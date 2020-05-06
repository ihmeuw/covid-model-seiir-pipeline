import pandas as pd
import yaml
from dataclasses import dataclass
import os
from typing import List, Dict

from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT, Directories, COVARIATE_COL_DICT


N_DRAWS = 1000


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


def load_covariates(directories, covariate_version, location_ids, draw_id=None):
    df = pd.read_csv(directories.get_cached_covariates_file(covariate_version, draw_id=draw_id))
    if location_ids is not None:
        assert isinstance(location_ids, List)
        df = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']].isin(location_ids)].copy()
    return df


@dataclass
class CovariateFormatter:

    directories: Directories
    covariate_draw_dict: Dict[str: bool]
    location_ids: List[int]

    def __post_init__(self):
        self.col_observed = COVARIATE_COL_DICT['COL_OBSERVED']
        self.col_loc_id = COVARIATE_COL_DICT['COL_LOC_ID']
        self.col_date = COVARIATE_COL_DICT['COL_DATE']

    def format_covariates(self, covariate_version, draw_id=None):
        dfs = pd.DataFrame()
        for name, pull_draws in self.covariate_draw_dict.items():
            df = pd.read_csv(self.directories.get_covariate_file(
                covariate_name=name, covariate_version=covariate_version
            ))
            if draw_id is not None:
                if pull_draws:
                    value_column = f'draw_{draw_id}'
                else:
                    value_column = name
            else:
                value_column = name
            df = df[[self.col_loc_id, self.col_date, value_column]]
            df = df.loc[~df[value_column].isnull()].copy()
            if dfs.empty:
                dfs = df
            else:
                # time dependent covariates versus not
                if self.col_date in df.columns:
                    dfs = dfs.merge(df, on=[self.col_loc_id, self.col_date])
                else:
                    dfs = dfs.merge(df, on=[self.col_loc_id])
            dfs = dfs.loc[dfs[self.col_loc_id].isin(self.location_ids)].copy()
        return dfs


def cache_covariates(directories, covariate_versions, location_ids, covariate_draw_dict):
    formatter = CovariateFormatter(
        directories=directories, covariate_draw_dict=covariate_draw_dict,
        location_ids=location_ids
    )
    pull_draws = covariate_draw_dict.values().any()
    for version in covariate_versions:
        if pull_draws:
            for draw_id in range(N_DRAWS):
                df = formatter.format_covariates(version, draw_id=draw_id)
                df.to_csv(directories.get_cached_covariates_file(covariate_version=version, draw_id=draw_id))
        else:
            df = formatter.format_covariates(version)
            df.to_csv(directories.get_cached_covariates_file(covariate_version=version))


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

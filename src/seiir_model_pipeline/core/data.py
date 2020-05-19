import pandas as pd
import yaml
from dataclasses import dataclass
import os
from typing import List, Dict
import warnings

from seiir_model_pipeline.core.versioner import Directories, COVARIATE_COL_DICT, COVARIATE_CACHE, COVARIATE_DIR
from seiir_model_pipeline.core.versioner import INPUT_DIR

N_DRAWS = 1000


def get_covariate_version_from_best():
    file = COVARIATE_DIR / 'best/metadata.yaml'
    with open(file) as f:
        version = yaml.load(f, Loader=yaml.FullLoader)
    path = version['output_path'].split('/')[-1]
    return path


def get_missing_locations(location_ids,
                          infection_version):
    infection_dir = INPUT_DIR / infection_version
    infection_loc = [x.split('_')[-1] for x in os.listdir(infection_dir)
                     if os.path.isdir(infection_dir / x)]
    infection_loc = [int(x) for x in infection_loc if x.isdigit()]
    missing_infection_loc = list()
    for loc in location_ids:
        if loc not in infection_loc:
            missing_infection_loc.append(loc)
    warnings.warn('Locations missing from infection data: ' + str(missing_infection_loc))

    return [x for x in location_ids if x not in infection_loc]


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


def load_ode_fits(directories, location_ids, draw_id):
    df_beta = []
    for l_id in location_ids:
        df_beta.append(pd.read_csv(directories.get_draw_beta_fit_file(l_id, draw_id)))
    df_beta = pd.concat(df_beta).reset_index()
    return df_beta


def load_covariates(directories, covariate_version, location_ids, draw_id=None):
    """
    Load covariates that have *already been cached*.

    :param directories: Directories object
    :param covariate_version: (str) covariate version
    :param location_ids:
    :param draw_id:
    :return:
    """
    df = pd.read_csv(directories.get_cached_covariates_file(covariate_version, draw_id=draw_id))
    if location_ids is not None:
        assert isinstance(location_ids, List)
        df = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']].isin(location_ids)].copy()
    return df


@dataclass
class CovariateFormatter:
    """
    Formats covariates by pulling in the files from the seir-covariates input directory.
    Deals with time dependent and independent covariates.
    """

    directories: Directories
    covariate_draw_dict: Dict[str, bool]
    location_ids: List[int]

    def __post_init__(self):
        self.col_observed = COVARIATE_COL_DICT['COL_OBSERVED']
        self.col_loc_id = COVARIATE_COL_DICT['COL_LOC_ID']
        self.col_date = COVARIATE_COL_DICT['COL_DATE']

    def format_covariates(self, covariate_version, draw_id=None):
        dfs = pd.DataFrame()
        value_columns = []
        for name, pull_draws in self.covariate_draw_dict.items():
            if name == 'intercept':
                continue
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
            value_columns.append(value_column)
            df = df.loc[~df[value_column].isnull()].copy()
            if dfs.empty:
                dfs = df
            else:
                # time dependent covariates versus not
                if self.col_date in df.columns:
                    dfs = dfs.merge(df, on=[self.col_loc_id, self.col_date])
                else:
                    dfs = dfs.merge(df, on=[self.col_loc_id])
        dfs = dfs[[self.col_loc_id, self.col_date] + value_columns]
        dfs = dfs.loc[dfs[self.col_loc_id].isin(self.location_ids)].copy()
        return dfs


def get_new_cache_version(covariate_version):
    """
    Create a cache version for covariates based on a covariate version.
    Also copies over the metadata so that we can access it later.

    :param covariate_version: (str) covariate version to read from seir-covariates
    :return: (str) cached version for seir-pipeline-outputs/covariates
    """
    if covariate_version == 'best':
        covariate_version = get_covariate_version_from_best()
    dirs = os.listdir(COVARIATE_CACHE)
    dirs = [x.split('.') for x in dirs]
    matched_versions = [x for x in dirs if '.'.join([x[0], x[1]]) == covariate_version]
    version = len(matched_versions) + 1
    new_version = f'{covariate_version}.{version:02}'
    os.makedirs(COVARIATE_CACHE / new_version)
    os.system(f'cp {str(COVARIATE_DIR / covariate_version / "metadata.yaml")} '
              f'{str(COVARIATE_CACHE / new_version / "metadata.yaml")}')
    return new_version


def cache_covariates(directories, covariate_version, location_ids, covariate_draw_dict):
    """
    Read in the covariates for this run, potentially per draw.

    :param directories: Directories object
    :param covariate_version: (str) the version of covariates to read
    :param location_ids: (List[int])
    :param covariate_draw_dict: (Dict[str, bool]) for each covariate, whether or not to read in draws or the mean
    :return: new version of covariates in the seir-pipeline-outputs directory that have been cached for later
    """
    cache_version = get_new_cache_version(covariate_version)
    formatter = CovariateFormatter(
        directories=directories, covariate_draw_dict=covariate_draw_dict,
        location_ids=location_ids
    )
    pull_draws = any(covariate_draw_dict.values())
    if pull_draws:
        for draw_id in range(N_DRAWS):
            df = formatter.format_covariates(covariate_version, draw_id=draw_id)
            df.to_csv(directories.get_cached_covariates_file(covariate_version=cache_version, draw_id=draw_id))
    else:
        df = formatter.format_covariates(covariate_version)
        df.to_csv(directories.get_cached_covariates_file(covariate_version=cache_version))

    return cache_version


def load_mr_coefficients(directories, draw_id):
    """
    Load meta-regression coefficients

    :param directories: Directories object
    :param draw_id: (int) which draw to load
    :return:
    """
    df = pd.read_csv(directories.get_draw_coefficient_file(draw_id))
    return df


def load_beta_fit(directories, draw_id, location_id):
    """
    Load the beta fit file for a draw / location.

    :param directories: Directories object
    :param draw_id: (int)
    :param location_id: (int)
    :return:
    """
    df = pd.read_csv(directories.get_draw_beta_fit_file(location_id, draw_id))
    return df


def load_beta_params(directories, draw_id):
    """
    Load the parameters that were sampled for this draw ID (alpha / gamma1 / gamma2 / sigma)

    :param directories: Directories object
    :param draw_id: (int)
    :return:
    """
    df = pd.read_csv(directories.get_draw_beta_param_file(draw_id))
    return df.set_index('params')['values'].to_dict()

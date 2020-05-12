import pandas as pd
import numpy as np

from seiir_model_pipeline.core.versioner import RegressionVersion, ForecastVersion, Directories
from seiir_model_pipeline.core.data import get_missing_locations
from seiir_model_pipeline.core.data import cache_covariates


def create_regression_version(version_name, covariate_version,
                              covariate_draw_dict,
                              infection_version,
                              location_set_version_id, **kwargs):
    """
    Utility function to create a regression version. Will cache covariates
    as well.

    :param version_name: (str) what do you want to name the version
    :param covariate_version: (str)
    :param covariate_draw_dict: (Dict[str, bool])
    :param infection_version: (str)
    :param location_set_version_id: (int)
    :param kwargs: other keyword arguments to a regression version.
    """
    directories = Directories()
    location_ids = get_locations(
        directories, infection_version,
        location_set_version_id=location_set_version_id,
    )
    cache_version = cache_covariates(
        directories=directories,
        covariate_version=covariate_version,
        location_ids=location_ids,
        covariate_draw_dict=covariate_draw_dict
    )
    rv = RegressionVersion(version_name=version_name, covariate_version=cache_version,
                           covariate_draw_dict=covariate_draw_dict,
                           location_set_version_id=location_set_version_id,
                           infection_version=infection_version, **kwargs)
    rv.create_version()
    rv_directory = Directories(regression_version=version_name)
    write_locations(directories=rv_directory, location_ids=location_ids)


def create_forecast_version(version_name, covariate_version,
                            covariate_draw_dict,
                            regression_version):
    """
    Utility function to create a regression version. Will cache covariates
    as well.

    :param version_name: (str) what do you want to name the version
    :param covariate_version: (str)
    :param covariate_draw_dict: (Dict[str, bool])
    :param regression_version: (str) which regression version to build off of
    """
    directories = Directories(regression_version=regression_version)
    location_ids = load_locations(directories)
    cache_version = cache_covariates(
        directories=directories,
        covariate_version=covariate_version,
        location_ids=location_ids,
        covariate_draw_dict=covariate_draw_dict
    )
    fv = ForecastVersion(version_name=version_name, covariate_version=cache_version,
                         regression_version=regression_version,
                         covariate_draw_dict=covariate_draw_dict)
    fv.create_version()


def create_run(version_name, covariate_version, covariate_draw_dict, **kwargs):
    """
    Creates a full run with a regression and a forecast version by the *SAME NAME*.
    :param version_name: (str) what will the name be for both regression and forecast versions
    :param covariate_version: (str) which covariate version to use
    :param covariate_draw_dict: (Dict[str, bool])
    :param kwargs: additional keyword arguments to regression version
    """
    create_regression_version(
        version_name=version_name, covariate_version=covariate_version,
        covariate_draw_dict=covariate_draw_dict,
        **kwargs
    )
    create_forecast_version(
        version_name=version_name, covariate_version=covariate_version,
        covariate_draw_dict=covariate_draw_dict,
        regression_version=version_name
    )
    print(f"Created regression and forecast versions {version_name}.")


def get_location_name_from_id(location_id, metadata_path):
    df = pd.read_csv(metadata_path)
    location_name = df.loc[df.location_id == location_id]['location_name'].iloc[0]
    return location_name


def date_to_days(date):
    date = pd.to_datetime(date)
    return np.array((date - date.min()).days)


def get_locations(directories, infection_version, location_set_version_id):
    df = pd.read_csv(
        directories.get_location_metadata_file(location_set_version_id),
    )
    missing = get_missing_locations(
        infection_version=infection_version,
        location_ids=df.location_id.unique().tolist()
    )
    locations = set(df.location_id.unique().tolist()) - set(missing)
    return list(locations)


def write_locations(directories, location_ids):
    df = pd.DataFrame({
        'location_id': location_ids
    })
    df.to_csv(directories.location_cache_file, index=False)


def load_locations(directories):
    return pd.read_csv(directories.location_cache_file).location_id.tolist()

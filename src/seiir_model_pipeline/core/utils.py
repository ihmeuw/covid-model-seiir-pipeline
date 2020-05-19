import pandas as pd
import numpy as np

from seiir_model_pipeline.core.versioner import ODEVersion, RegressionVersion, ForecastVersion, Directories
from seiir_model_pipeline.core.data import get_missing_locations
from seiir_model_pipeline.core.data import cache_covariates


def create_ode_version(version_name, infection_version, location_set_version_id, **kwargs):
    """
    Utility function to create an ODE version.

    :param version_name: (str) what do you want to name the version
    :param infection_version: (str)
    :param location_set_version_id: (int)
    :param kwargs: other keyword arguments to an ode version
    """
    directories = Directories()
    location_ids = get_locations(
        directories, infection_version,
        location_set_version_id=location_set_version_id,
    )
    ov = ODEVersion(version_name=version_name, location_set_version_id=location_set_version_id,
                    infection_version=infection_version, **kwargs)
    ov.create_version()

    ov_directory = Directories(ode_version=version_name)
    write_locations(directories=ov_directory, location_ids=location_ids)


def create_regression_version(version_name, ode_version, covariate_version,
                              covariate_draw_dict, **kwargs):
    """
    Utility function to create a regression version. Will cache covariates
    as well.

    :param version_name: (str) what do you want to name the version
    :param ode_version: (str) what is the linked ode version
    :param covariate_version: (str)
    :param covariate_draw_dict: (Dict[str, bool])
    :param kwargs: other keyword arguments to a regression version.
    """
    directories = Directories(ode_version=ode_version)
    location_ids = load_locations(directories)

    cache_version = cache_covariates(
        directories=directories,
        covariate_version=covariate_version,
        location_ids=location_ids,
        covariate_draw_dict=covariate_draw_dict
    )
    rv = RegressionVersion(version_name=version_name, covariate_version=cache_version,
                           covariate_draw_dict=covariate_draw_dict, **kwargs)
    rv.create_version()


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


def create_run(version_name, covariate_version, covariate_draw_dict,
               covariates_order, coefficient_version=None, **kwargs):
    """
    Creates a full run with an ODE, regression and forecast version by the *SAME NAME*.

    - `version_name (str)`: what will the name be for both regression and forecast versions
    - `covariate_version (str)`: the version of the covariate inputs
    - `coefficient_version (str)`: the regression version of coefficient estimates to use
    - `covariates (Dict[str: Dict]): elements of the inner dict:
        - "use_re": (bool)
        - "gprior": (np.array)
        - "bounds": (np.array)
        - "re_var": (float)
    - `covariates_order (List[List[str]])`: list of lists of covariate names that will be
        sequentially added to the regression
    - `covariate_draw_dict (Dict[str, bool[)`: whether or not to use draws of the covariate (they
        must be available!)
    - `kwargs`: additional keyword arguments to regression version
    """
    create_ode_version(
        version_name=version_name, **kwargs
    )
    create_regression_version(
        version_name=version_name,
        covariate_version=covariate_version,
        covariate_draw_dict=covariate_draw_dict, covariates_order=covariates_order,
        coefficient_version=coefficient_version,
        ode_version=version_name
    )
    create_forecast_version(
        version_name=version_name, covariate_version=covariate_version,
        covariate_draw_dict=covariate_draw_dict,
        regression_version=version_name
    )
    print(f"Created ode, regression and forecast versions {version_name}.")


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

from typing import Dict, Tuple, Union
import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.core.versioner import RegressionVersion, ForecastVersion, Directories
from covid_model_seiir_pipeline.core.data import get_missing_locations
from covid_model_seiir_pipeline.core.data import cache_covariates


def create_regression_version(version_name, covariate_version,
                              covariate_draw_dict,
                              infection_version,
                              location_set_version_id, theta_config=dict(), **kwargs):
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
                           infection_version=infection_version, 
                           **theta_config, **kwargs)
    rv.create_version()
    rv_directory = Directories(regression_version=version_name)
    write_locations(directories=rv_directory, location_ids=location_ids)


def create_forecast_version(version_name,
                            covariate_version,
                            covariate_draw_dict,
                            regression_version):
    """
    Utility function to create a regression version. Will cache covariates
    as well.

    :param version_name: (str) what do you want to name the version
    :param covariate_version: (str)
    :param covariate_draw_dict: (Dict[str, bool])
    :param regression_version: (str) which regression version to build off of
    :param theta_config: (dict, optional) configuration for adding or removing
        people from the I1 bin of the ode.

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


def create_run(version_name, covariate_version, covariate_draw_dict, theta_config=None, **kwargs):
    """
    Creates a full run with a regression and a forecast version by the *SAME NAME*.
    :param version_name: (str) what will the name be for both regression and forecast versions
    :param covariate_version: (str) which covariate version to use
    :param covariate_draw_dict: (Dict[str, bool])
    :param kwargs: additional keyword arguments to regression version
    :param theta_config: (int, optional) configuration for adding or removing
        people from the I1 bin of the ode.
    """
    create_regression_version(
        version_name=version_name,
        covariate_version=covariate_version,
        covariate_draw_dict=covariate_draw_dict,
        theta_config=theta_config,
        **kwargs
    )
    create_forecast_version(
        version_name=version_name,
        covariate_version=covariate_version,
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


def beta_shift(beta_fit: pd.DataFrame,
               beta_pred: np.ndarray,
               draw_id: int,
               window_size: Union[int, None] = None,
               average_over_min: int = 1,
               average_over_max: int = 35) -> Tuple[np.ndarray, Dict[str, float]]:
    """Calculate the beta shift.

    Args:
        beta_fit (pd.DataFrame): Data frame contains the date and beta fit.
        beta_pred (np.ndarray): beta prediction.
        draw_id (int): Draw of data provided.  Will be used as a seed for
            a random number generator to determine the amount of beta history
            to leverage in rescaling the y-intercept for the beta prediction.
        window_size (Union[int, None], optional):
            Window size for the transition. If `None`, Hard shift no transition.
            Default to None.

    Returns:
        Tuple[np.ndarray, float]: Predicted beta, after scaling (shift) and the initial scaling.
    """
    assert 'date' in beta_fit.columns, "'date' has to be in beta_fit data frame."
    assert 'beta' in beta_fit.columns, "'beta' has to be in beta_fit data frame."
    beta_fit = beta_fit.sort_values('date')
    beta_hat = beta_fit['beta_pred'].to_numpy()
    beta_fit = beta_fit['beta'].to_numpy()

    rs = np.random.RandomState(seed=draw_id)
    avg_over = rs.randint(average_over_min, average_over_max)

    beta_fit_final = beta_fit[-1]
    beta_pred_start = beta_pred[0]

    scale_init = beta_fit_final / beta_pred_start
    log_beta_resid = np.log(beta_fit / beta_hat)
    scale_final = np.exp(log_beta_resid[-avg_over:].mean())

    scale_params = {
        'window_size': window_size,
        'history_days': avg_over,
        'fit_final': beta_fit_final,
        'pred_start': beta_pred_start,
        'beta_ratio_mean': scale_final,
        'beta_residual_mean': np.log(scale_final),
    }

    if window_size is not None:
        assert isinstance(window_size, int) and window_size > 0, f"window_size={window_size} has to be a positive " \
                                                                 f"integer."
        scale = scale_init + (scale_final - scale_init)/window_size*np.arange(beta_pred.size)
        scale[(window_size + 1):] = scale_final
    else:
        scale = scale_init

    betas = beta_pred * scale

    return betas, scale_params

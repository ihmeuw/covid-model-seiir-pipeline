from pathlib import Path
import warnings
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Union, Tuple
import os
import json
import numpy as np


BASE_DIR = Path('/ihme')

# Dependency/input directories
INPUT_DIR = BASE_DIR / 'covid-19/seir-inputs'
COVARIATE_DIR = BASE_DIR / 'covid-19/seir-covariates'

# Output directory
OUTPUT_DIR = BASE_DIR / 'covid-19/seir-pipeline-outputs'

# Four main directories in the outputs directory
METADATA_DIR = OUTPUT_DIR / 'metadata-inputs'
REGRESSION_OUTPUT = OUTPUT_DIR / 'regression'
FORECAST_OUTPUT = OUTPUT_DIR / 'forecast'
COVARIATE_CACHE = OUTPUT_DIR / 'covariate'

# File pattern for storing infectionator outputs
INFECTION_FILE_PATTERN = 'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv'

# Where location metadata is stored
LOCATION_METADATA_FILE_PATTERN = 'location_metadata_{lsvid}.csv'
# This is a list of locations used for a particular run
LOCATION_CACHE_FILE = 'locations.csv'

# Where cached covariates are stored
CACHED_COVARIATES_FILE = 'cached_covariates.csv'
CACHED_COVARIATES_DRAW_FILE = 'cached_covariates_draw_{draw_id}.csv'

# Columns from infectionator inputs
INFECTION_COL_DICT = {
    'COL_DATE': 'date',
    'COL_CASES': 'cases_draw',
    'COL_POP': 'pop',
    'COL_LOC_ID': 'loc_id',
    'COL_DEATHS': 'deaths_draw',
    'COL_ID_LAG': 'i_d_lag',
    'COL_OBS_DEATHS': 'obs_deaths',
    'COL_OBS_CASES': 'obs_infecs',
    'COL_DEATHS_DATA': 'deaths_mean'
}

# Columns from covariates inputs
COVARIATE_COL_DICT = {
    'COL_DATE': 'date',
    'COL_OBSERVED': 'observed',
    'COL_LOC_ID': 'location_id'
}

# The key for the observed column
OBSERVED_DICT = {
    'observed': 1.,
    'forecasted': 0.
}


class FileDoesNotExist(Exception):
    pass


def _get_infection_folder_from_location_id(location_id, input_dir):
    """
    This is the infection input folder. It's a helper because
    folders have location names in them.

    :param location_id: (int)
    :param input_dir: (Path)
    """
    folders = os.listdir(input_dir)
    correct = np.array([f.endswith(f'_{location_id}') for f in folders])
    if correct.sum() > 1:
        raise RuntimeError(f"There is more than one location-specific folder for {location_id}.")
    elif correct.sum() == 0:
        raise FileDoesNotExist(f"There is not a location-specific folder for {location_id}.")
    else:
        pass
    folder = folders[np.where(correct)[0][0]]
    return folder


def args_to_directories(args):
    """

    :param args: result of an argparse.ArgumentParser.parse_args()
    :return: (Directories) object
    """
    return Directories(
        regression_version=args.regression_version,
        forecast_version=args.forecast_version
    )


@dataclass
class Directories:
    """
    Directories object. You can use the Directories object by itself, or you can pass
    in a regression and/or forecast version to get files/paths specific to those
    parts of the pipeline.

    - `regression_version (str)`: regression version
    - `forecast_version (str)`: forecast version
    """
    regression_version: str = None
    forecast_version: str = None

    def __post_init__(self):
        rv = None
        fv = None
        if self.regression_version is None:
            if self.forecast_version is None:
                pass
            else:
                fv = load_forecast_settings(self.forecast_version)
                rv = load_regression_settings(fv.regression_version)
        else:
            rv = load_regression_settings(self.regression_version)
            if self.forecast_version is not None:
                fv = load_forecast_settings(self.forecast_version)
                check_compatible_version(rv, fv)

        if rv is not None:
            self.infection_dir = INPUT_DIR / rv.infection_version
            self.rv_covariate_input_dir = COVARIATE_DIR / rv.covariate_version
            self.rv_covariate_cache_dir = COVARIATE_CACHE / rv.covariate_version

            self.regression_output_dir = REGRESSION_OUTPUT / rv.version_name

            self.regression_beta_fit_dir = self.regression_output_dir / 'betas'
            self.regression_parameters_dir = self.regression_output_dir / 'parameters'
            self.regression_coefficient_dir = self.regression_output_dir / 'coefficients'
            self.regression_diagnostic_dir = self.regression_output_dir / 'diagnostics'

        else:
            self.infection_dir = None
            self.rv_covariate_input_dir = None
            self.rv_covariate_cache_dir = None
            self.regression_output_dir = None
            self.regression_beta_fit_dir = None
            self.regression_parameters_dir = None
            self.regression_coefficient_dir = None
            self.regression_diagnostic_dir = None

        if fv is not None:
            self.forecast_output_dir = FORECAST_OUTPUT / fv.version_name
            self.fv_covariate_input_dir = COVARIATE_DIR / fv.covariate_version
            self.fv_covariate_cache_dir = COVARIATE_CACHE / fv.covariate_version

            self.forecast_component_draw_dir = self.forecast_output_dir / 'component_draws'
            self.forecast_output_draw_dir = self.forecast_output_dir / 'output_draws'
            self.forecast_diagnostic_dir = self.forecast_output_dir / 'diagnostics'
            self.forecast_beta_scaling_dir = self.forecast_output_dir / 'beta_scaling'
        else:
            self.forecast_output_dir = None

            self.fv_covariate_input_dir = None
            self.fv_covariate_cache_dir = None

            self.forecast_component_draw_dir = None
            self.forecast_output_draw_dir = None
            self.forecast_diagnostic_dir = None
            self.forecast_beta_scaling_dir = None

    def make_dirs(self):
        for directory in [
            self.regression_output_dir, self.forecast_output_dir,
            self.regression_coefficient_dir, self.regression_diagnostic_dir,
            self.regression_beta_fit_dir, self.regression_parameters_dir,
            self.forecast_diagnostic_dir, self.forecast_output_dir,
            self.forecast_component_draw_dir, self.forecast_output_draw_dir,
            self.forecast_beta_scaling_dir,
            self.rv_covariate_cache_dir, self.fv_covariate_cache_dir
        ]:
            if directory is not None:
                os.makedirs(str(directory), exist_ok=True)

    def get_draw_beta_fit_file(self, location_id, draw_id):
        os.makedirs(self.regression_beta_fit_dir / str(location_id), exist_ok=True)
        return self.regression_beta_fit_dir / str(location_id) / f'fit_draw_{draw_id}.csv'

    def get_draw_beta_param_file(self, draw_id):
        return self.regression_parameters_dir / f'params_draw_{draw_id}.csv'

    def get_draw_coefficient_file(self, draw_id):
        return self.regression_coefficient_dir / f'coefficients_{draw_id}.csv'

    def location_draw_component_forecast_file(self, location_id, draw_id):
        os.makedirs(self.forecast_component_draw_dir / str(location_id), exist_ok=True)
        return self.forecast_component_draw_dir / str(location_id) / f'draw_{draw_id}.csv'

    def location_output_forecast_file(self, location_id, forecast_type):
        if forecast_type not in ['deaths', 'cases', 'reff']:
            raise RuntimeError("Unrecognized forecast type.")
        return self.forecast_output_draw_dir / f'{forecast_type}_{location_id}.csv'

    def location_beta_scaling_file(self, location_id):
        return self.forecast_beta_scaling_dir / f'{location_id}_beta_scaling.csv'

    def get_infection_file(self, location_id, draw_id):
        folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        return self.infection_dir / folder / INFECTION_FILE_PATTERN.format(draw_id=draw_id)

    @staticmethod
    def get_covariate_file(covariate_version, covariate_name):
        return COVARIATE_DIR / covariate_version / f'{covariate_name}.csv'

    @staticmethod
    def get_cached_covariates_file(covariate_version, draw_id=None):
        if draw_id is None:
            return COVARIATE_CACHE / covariate_version / CACHED_COVARIATES_FILE
        else:
            return COVARIATE_CACHE / covariate_version / CACHED_COVARIATES_DRAW_FILE.format(draw_id=draw_id)

    @staticmethod
    def get_location_metadata_file(location_set_version_id):
        return METADATA_DIR / LOCATION_METADATA_FILE_PATTERN.format(lsvid=location_set_version_id)

    @property
    def location_cache_file(self):
        return self.regression_output_dir / LOCATION_CACHE_FILE


class VersionAlreadyExists(RuntimeError):
    pass


class VersionDoesNotExist(RuntimeError):
    pass


def _available_version(version):
    if os.path.exists(version):
        raise VersionAlreadyExists


def _confirm_version_exists(version):
    if not os.path.exists(version):
        raise VersionDoesNotExist


def _get_regression_settings_file(regression_version):
    return REGRESSION_OUTPUT / str(regression_version) / 'settings.json'


def _get_forecast_settings_file(forecast_version):
    return FORECAST_OUTPUT / str(forecast_version) / 'settings.json'


def load_regression_settings(regression_version):
    file = _get_regression_settings_file(regression_version)
    with open(file, 'r') as f:
        settings = json.load(f)
        if isinstance(settings['day_shift'], int):
            day_shift = settings['day_shift']
            settings.update({'day_shift': [day_shift, day_shift]})
    return RegressionVersion(**settings)


def load_forecast_settings(forecast_version):
    file = _get_forecast_settings_file(forecast_version)
    with open(file, 'r') as f:
        settings = json.load(f)
    return ForecastVersion(**settings)


def check_compatible_version(regression_version, forecast_version):
    """
    Checks that two versions can be compatible -- regression and forecast.
    For example, you can't have covariates that are in a forecast version that weren't
    in a regression version or vice versa.

    :param regression_version: (str)
    :param forecast_version: (str)
    :return:
    """
    assert regression_version.covariate_draw_dict.keys() == forecast_version.covariate_draw_dict.keys()
    if regression_version.covariate_version == forecast_version.covariate_version:
        for covariate, draw in regression_version.covariate_draw_dict.items():
            assert covariate in forecast_version.covariate_draw_dict
            if draw and not forecast_version.covariate_draw_dict[covariate]:
                raise RuntimeError("Incompatible regression and forecast covariate settings.")
            if not draw and forecast_version.covariate_draw_dict[covariate]:
                raise RuntimeError("Incompatible regression and forecast covariate settings.")


@dataclass
class Version:
    """
    - `version_name (str)`: the name of the output version
    """

    version_name: str


@dataclass
class RegressionVersion(Version):
    """
    A regression version is the first half of the SEIIR pipeline. It fits a spline to newE,
    runs an ODE, and then does a regression of beta ~ covariates and saves those coefficients
    for the forecasting component.

    - `infection_version (str)`: the version of the infection inputs
    - `covariate_version (str)`: the version of the covariate inputs
    - `coefficient_version (str)`: the regression version of coefficient estimates to use
    - `n_draws (int)`: number of draws
    - `location_set_version_id (int)`: the location set version to use
    - `degree` (int): degree of the spline for beta fit
    - `knots` (int)`: knot positions for the spline
    - `day_shift (Tuple[int])`: Will use today + `day_shift` - lag 's data in the beta regression
        but will sample this day_shift from the range given
    - `covariates (Dict[str: Dict]): elements of the inner dict:
        - "use_re": (bool)
        - "gprior": (np.array)
        - "bounds": (np.array)
        - "re_var": (float)
    - `covariates_order (List[List[str]])`: list of lists of covariate names that will be
        sequentially added to the regression
    - `covariate_draw_dict (Dict[str, bool[)`: whether or not to use draws of the covariate (they
        must be available!)
    - `sequential` (bool): should the regression be fit sequentialy according to the
        ordering of `covariates_order`? If False, the ordering of `covariates_order`
        does not matter for 
    - `alpha (Tuple[float])`: a 2-length tuple that represents the range of alpha values to sample
    - `sigma (Tuple[float])`: a 2-length tuple that represents the range of sigma values to sample
    - `gamma1 (Tuple[float])`: a 2-length tuple that represents the range of gamma1 values to sample
    - `gamma2 (Tuple[float])`: a 2-length tuple that represents the range of gamma2 values to sample
    - `solver_dt (float)`: step size for the ODE solver
    """

    infection_version: str
    covariate_version: str

    n_draws: int
    location_set_version_id: int

    # Spline required arguments
    degree: int
    knots: np.array

    # Regression Arguments
    covariates: Dict[str, Dict[str, Union[bool, List, float]]]
    covariates_order: List[List[str]] = None
    covariate_draw_dict: Dict[str, bool] = None
    sequential: bool = False

    coefficient_version: str = None

    # Spline optional arguments
    concavity: bool = True
    increasing: bool = False
    spline_se_power: float = field(default=1.0)
    spline_space: str = field(default='ln daily')
    spline_knots_type: str = field(default='domain')
    spline_r_linear: bool = True
    spline_l_linear: bool = True

    day_shift: Tuple[int] = field(default=(0, 8))

    # Optimization Arguments
    alpha: Tuple[float] = field(default=(0.95, 0.95))
    sigma: Tuple[float] = field(default=(0.20, 0.20))
    gamma1: Tuple[float] = field(default=(0.50, 0.50))
    gamma2: Tuple[float] = field(default=(0.50, 0.50))
    solver_dt: float = field(default=0.1)

    beta_shift_dict: Dict = field(default_factory=lambda :dict(window_size=None))

    def __post_init__(self):
        if not self.spline_space.startswith('ln') and self.spline_se_power !=0.0:
            warnings.warn("Spline not fitting in the log space, spline_se_power advised to be 0.", UserWarning)

        if self.spline_space.endswith('daily') and self.increasing:
            warnings.warn("Spline fitting daily data, do not suggest using increasing constraint.", UserWarning)

        if self.spline_space.endswith('cumul') and self.concavity:
            warnings.warn("Spline fitting cumulative data, do not suggest using concave constraint.", UserWarning)

    def _settings_to_json(self):
        settings = asdict(self)
        _confirm_version_exists(REGRESSION_OUTPUT / self.version_name)
        file = _get_regression_settings_file(self.version_name)
        with open(file, "w") as f:
            json.dump(settings, f)

    def create_version(self):
        _available_version(REGRESSION_OUTPUT / self.version_name)
        os.makedirs(REGRESSION_OUTPUT / self.version_name)
        self._settings_to_json()
        return self.version_name


@dataclass
class ForecastVersion(Version):
    """
    A forecast version reads in a regression version's output and forecasts the beta forwards
    and then runs an ODE forward to get all of the compartments.

    - `regression_version (str)`: the regression version to read from
    - `covariate_version (str)`: the covariate version to pull
    - `covariate_draw_dict (Dict[str, bool[)`: whether or not to use draws of the covariate (they
        must be available!)
    """

    regression_version: str
    covariate_version: str

    covariate_draw_dict: Dict[str, bool]
    theta: float = None

    def __post_init__(self):
        pass

    def _settings_to_json(self):
        settings = asdict(self)
        _confirm_version_exists(FORECAST_OUTPUT / self.version_name)
        file = _get_forecast_settings_file(self.version_name)
        with open(file, "w") as f:
            json.dump(settings, f)

    def create_version(self):
        _available_version(FORECAST_OUTPUT / self.version_name)
        os.makedirs(FORECAST_OUTPUT / self.version_name)
        self._settings_to_json()
        return self.version_name

from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Union, Tuple
import os
import json
import numpy as np


BASE_DIR = Path('/ihme/covid-19')

# Dependency/input directories
INPUT_DIR = BASE_DIR / 'seir-inputs'
COVARIATE_DIR = BASE_DIR / 'seir-covariates'

# Output directory
OUTPUT_DIR = BASE_DIR / 'seir-pipeline-outputs'

# Five main directories in the outputs directory
METADATA_DIR = OUTPUT_DIR / 'metadata-inputs'
COVARIATE_CACHE = OUTPUT_DIR / 'covariate'

ODE_OUTPUT = OUTPUT_DIR / 'ode'
REGRESSION_OUTPUT = OUTPUT_DIR / 'regression'
FORECAST_OUTPUT = OUTPUT_DIR / 'forecast'

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
        ode_version=None if not hasattr(args, 'ode_version') else args.ode_version,
        regression_version=None if not hasattr(args, 'regression_version') else args.regression_version,
        forecast_version=None if not hasattr(args, 'forecast_version') else args.forecast_version
    )


def make_dirs(dir_list):
    for directory in dir_list:
        if directory is not None:
            os.makedirs(directory, exist_ok=True)


@dataclass
class Directories:
    """
    Directories object. You can use the Directories object by itself, or you can pass
    in an ode and/or regression and/or forecast version to get files/paths specific to those
    parts of the pipeline.

    - `ode_version (str)`: beta ODE fit version
    - `regression_version (str)`: regression version
    - `forecast_version (str)`: forecast version
    """
    ode_version: Union[str, None] = None
    regression_version: Union[str, None] = None
    forecast_version: Union[str, None] = None

    def __post_init__(self):
        ov = None
        rv = None
        fv = None

        self.ode_output_dir = None
        self.regression_output_dir = None
        self.forecast_output_dir = None

        self.infection_dir = None
        self.beta_fit_dir = None

        self.rv_covariate_input_dir = None
        self.rv_covariate_cache_dir = None

        self.regression_parameters_dir = None
        self.regression_coefficient_dir = None
        self.regression_diagnostic_dir = None

        self.fv_covariate_input_dir = None
        self.fv_covariate_cache_dir = None

        self.forecast_component_draw_dir = None
        self.forecast_output_draw_dir = None
        self.forecast_diagnostic_dir = None
        self.forecast_beta_scaling_dir = None
        
        if self.ode_version is not None:
            ov = load_ode_settings(self.ode_version)
        if self.regression_version is not None:
            rv = load_regression_settings(self.regression_version)
            ov = load_ode_settings(rv.ode_version)
        if self.forecast_version is not None:
            fv = load_forecast_settings(self.forecast_version)
            rv = load_regression_settings(fv.regression_version)
            ov = load_ode_settings(rv.ode_version)
        
        self.ode_version = ov.version_name if ov is not None else None
        self.regression_version = rv.version_name if rv is not None else None
        self.forecast_version = fv.version_name if fv is not None else None

        if ov is not None:
            self.ode_output_dir = ODE_OUTPUT / ov.version_name
            self.ode_beta_fit_dir = self.ode_output_dir / 'betas'
            self.ode_parameters_dir = self.ode_output_dir / 'parameters'

            self.infection_dir = Path(ov.infection_dir) / ov.infection_version

            make_dirs([
                self.ode_output_dir,
                self.infection_dir,
                self.ode_beta_fit_dir,
                self.ode_parameters_dir
            ])

        if rv is not None:
            self.regression_output_dir = REGRESSION_OUTPUT / rv.version_name

            self.rv_covariate_input_dir = COVARIATE_DIR / rv.covariate_version
            self.rv_covariate_cache_dir = COVARIATE_CACHE / rv.covariate_version

            self.regression_betas_dir = self.regression_output_dir / 'betas'
            self.regression_coefficient_dir = self.regression_output_dir / 'coefficients'
            self.regression_diagnostic_dir = self.regression_output_dir / 'diagnostics'

            make_dirs([
                self.regression_output_dir,
                self.rv_covariate_input_dir,
                self.rv_covariate_cache_dir,
                self.regression_coefficient_dir,
                self.regression_diagnostic_dir
            ])

        if fv is not None:
            self.forecast_output_dir = FORECAST_OUTPUT / fv.version_name

            self.fv_covariate_input_dir = COVARIATE_DIR / fv.covariate_version
            self.fv_covariate_cache_dir = COVARIATE_CACHE / fv.covariate_version

            self.forecast_component_draw_dir = self.forecast_output_dir / 'component_draws'
            self.forecast_output_draw_dir = self.forecast_output_dir / 'output_draws'
            self.forecast_diagnostic_dir = self.forecast_output_dir / 'diagnostics'
            self.forecast_beta_scaling_dir = self.forecast_output_dir / 'beta_scaling'

            make_dirs([
                self.forecast_output_dir,
                self.fv_covariate_input_dir,
                self.fv_covariate_cache_dir,
                self.forecast_component_draw_dir,
                self.forecast_output_draw_dir,
                self.forecast_diagnostic_dir,
                self.forecast_beta_scaling_dir
            ])

    def get_draw_beta_fit_file(self, location_id, draw_id):
        os.makedirs(self.ode_beta_fit_dir / str(location_id), exist_ok=True)
        return self.ode_beta_fit_dir / str(location_id) / f'fit_draw_{draw_id}.csv'

    def get_draw_beta_regression_file(self, location_id, draw_id):
        os.makedirs(self.regression_betas_dir / str(location_id), exist_ok=True)
        return self.regression_betas_dir / str(location_id) / f'regression_draw_{draw_id}.csv'

    def get_draw_beta_param_file(self, draw_id):
        return self.ode_parameters_dir / f'params_draw_{draw_id}.csv'

    def get_draw_coefficient_file(self, draw_id, regression_version=None):
        if regression_version is None:
            return self.regression_coefficient_dir / f'coefficients_{draw_id}.csv'
        else:
            assert type(regression_version) == str, "pass in a regression version name as string"
            return REGRESSION_OUTPUT / regression_version / 'coefficients' / f'coefficients_{draw_id}.csv'

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
        return self.ode_output_dir / LOCATION_CACHE_FILE


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


def _get_ode_settings_file(ode_version):
    return ODE_OUTPUT / str(ode_version) / 'settings.json'


def _get_regression_settings_file(regression_version):
    return REGRESSION_OUTPUT / str(regression_version) / 'settings.json'


def _get_forecast_settings_file(forecast_version):
    return FORECAST_OUTPUT / str(forecast_version) / 'settings.json'


def load_ode_settings(ode_version):
    file = _get_ode_settings_file(ode_version)
    with open(file, 'r') as f:
        settings = json.load(f)
        if isinstance(settings['day_shift'], int):
            day_shift = settings['day_shift']
            settings.update({'day_shift': [day_shift, day_shift]})
    return ODEVersion(**settings)


def load_regression_settings(regression_version):
    file = _get_regression_settings_file(regression_version)
    with open(file, 'r') as f:
        settings = json.load(f)
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
class ODEVersion(Version):
    """
    An ODE version is the first step in the SEIIR pipeline. It fits a spline to newE, and
    runs an ODE.

    - `infection_version (str)`: the version of the infection inputs
    - `n_draws (int)`: number of draws
    - `location_set_version_id (int)`: the location set version to use
    - `degree` (int): degree of the spline for beta fit
    - `knots` (int)`: knot positions for the spline
    - `day_shift (Tuple[int])`: Will use today + `day_shift` - lag 's data in the beta regression
        but will sample this day_shift from the range given
    - `alpha (Tuple[float])`: a 2-length tuple that represents the range of alpha values to sample
    - `sigma (Tuple[float])`: a 2-length tuple that represents the range of sigma values to sample
    - `gamma1 (Tuple[float])`: a 2-length tuple that represents the range of gamma1 values to sample
    - `gamma2 (Tuple[float])`: a 2-length tuple that represents the range of gamma2 values to sample
    - `solver_dt (float)`: step size for the ODE solver
    """
    infection_version: str
    n_draws: int
    location_set_version_id: int

    # Spline Arguments
    degree: int
    knots: np.array

    day_shift: Tuple[int] = field(default=(0, 8))

    # Optimization Arguments
    alpha: Tuple[float] = field(default=(0.95, 0.95))
    sigma: Tuple[float] = field(default=(0.20, 0.20))
    gamma1: Tuple[float] = field(default=(0.50, 0.50))
    gamma2: Tuple[float] = field(default=(0.50, 0.50))
    solver_dt: float = field(default=0.1)

    infection_dir: str = INPUT_DIR

    def __post_init__(self):
        pass

    def _settings_to_json(self):
        settings = asdict(self)
        _confirm_version_exists(ODE_OUTPUT / self.version_name)
        file = _get_ode_settings_file(self.version_name)
        with open(file, "w") as f:
            json.dump(settings, f)

    def create_version(self):
        _available_version(ODE_OUTPUT / self.version_name)
        os.makedirs(ODE_OUTPUT / self.version_name)
        self._settings_to_json()
        return self.version_name


@dataclass
class RegressionVersion(Version):
    """
    A regression version is the first half of the SEIIR pipeline.
    It does a regression of beta ~ covariates and saves those coefficients
    for the forecasting component.

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
    """
    ode_version: str
    covariate_version: str

    # Regression Arguments
    covariates: Dict[str, Dict[str, Union[bool, List, float]]]
    covariates_order: List[List[str]] = None
    covariate_draw_dict: Dict[str, bool] = None

    coefficient_version: str = None

    def __post_init__(self):
        pass

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

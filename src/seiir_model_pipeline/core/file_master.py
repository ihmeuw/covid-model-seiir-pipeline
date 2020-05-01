from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Union
import os
import json
import numpy as np

BASE_DIR = Path('/ihme')

# Dependency directories
INPUT_DIR = BASE_DIR / 'covid-19/seir-inputs'
COVARIATE_DIR = BASE_DIR / 'covid-19/seir-covariates'

# Output directories
OUTPUT_DIR = BASE_DIR / 'scratch/users/mnorwood/covid'
REGRESSION_OUTPUT = OUTPUT_DIR / 'regression'
FORECAST_OUTPUT = OUTPUT_DIR / 'forecast'

INFECTION_FILE_PATTERN = 'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv'
PEAK_DATE_FILE = '/ihme/scratch/projects/covid/seir_research_test_run/death_model_peaks.csv'

PEAK_DATE_COL_DICT = {
    'COL_LOC_ID': 'location_id',
    'COL_DATE': 'peak_date'
}

INFECTION_COL_DICT = {
    'COL_DATE': 'date',
    'COL_CASES': 'cases',
    'COL_POP': 'pop',
    'COL_LOC_ID': 'loc_id'
}

COVARIATE_COL_DICT = {
    'COL_DATE': 'date',
    'COL_OBSERVED': 'observed',
    'COL_LOC_ID': 'location_id'
}


def _get_infection_folder_from_location_id(location_id, input_dir):
    """
    This is the infection input folder. It's a helper because
    folders have location names in them.

    :param location_id: (int)
    :param input_dir: (Path)
    :return: (str)
    """
    folders = os.listdir(input_dir)
    correct = np.array([f.endswith(f'_{location_id}') for f in folders])
    if correct.sum() > 1:
        raise RuntimeError(f"There is more than one location-specific folder for {location_id}.")
    elif correct.sum() == 0:
        raise RuntimeError(f"There is not a location-specific folder for {location_id}.")
    else:
        pass
    folder = folders[np.where(correct)[0][0]]
    return folder


def _get_loc_scenario_draw_file(location_id, draw_id, scenario_id, directory):
    """
    This is the location-scenario-draw file.

    :param location_id: (int)
    :param draw_id: (int)
    :param scenario_id: (int)
    :param directory: (Path) parent directory
    :return: (Path)
    """
    return directory / f'{location_id}/draw{draw_id}_scenario{scenario_id}.csv'


def _get_loc_scenario_file(location_id, scenario_id, directory):
    """
    This is the final location-scenario file with all draws.

    :param location_id: (int)
    :param scenario_id: (int)
    :param directory: (Path) parent directory
    :return: (Path)
    """
    return directory / f'{location_id}/scenario{scenario_id}.csv'


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
    ## Arguments

    - `infection_version (str)`: version of the infections to pull
    - `covariate_version (str)`: version of the covariates to pull
    - `output_version (str)`: version of outputs to store
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

        if rv is not None:
            self.infection_dir = INPUT_DIR / rv.infection_version
            self.covariate_dir = COVARIATE_DIR / rv.covariate_version

            self.regression_output_dir = REGRESSION_OUTPUT / rv.version_name

            self.regression_coefficient_dir = self.regression_output_dir / 'coefficients'
            self.regression_diagnostic_dir = self.regression_output_dir / 'diagnostics'
        else:
            self.infection_dir = None
            self.covariate_dir = None
            self.regression_output_dir = None
            self.regression_coefficient_dir = None
            self.regression_diagnostic_dir = None

        if fv is not None:
            self.forecast_output_dir = FORECAST_OUTPUT / fv.version_name

            self.forecast_draw_dir = self.forecast_output_dir / 'location_draws'
            self.forecast_diagnostic_dir = self.forecast_output_dir / 'diagnostics'
        else:
            self.forecast_output_dir = None
            self.forecast_draw_dir = None
            self.forecast_diagnostic_dir = None

        self.log_dir = OUTPUT_DIR / 'logs'

    def make_dirs(self):
        for directory in [
            self.regression_output_dir, self.forecast_output_dir,
            self.regression_coefficient_dir, self.regression_diagnostic_dir,
            self.forecast_draw_dir, self.forecast_diagnostic_dir,
            self.log_dir
        ]:
            if directory is not None:
                os.makedirs(str(directory), exist_ok=True)

    def draw_ode_fit_file(self, draw_id):
        return self.regression_output_dir / f'fit_draw_{draw_id}.csv'

    def draw_ode_param_file(self, draw_id):
        return self.regression_output_dir / f'params_draw_{draw_id}.csv'

    def location_draw_forecast_file(self, location_id, draw_id):
        os.makedirs(self.forecast_output_dir / str(location_id), exist_ok=True)
        return self.forecast_output_dir / str(location_id) / f'draw_{draw_id}.csv'

    def get_infection_file(self, location_id, draw_id):
        folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        return self.infection_dir / folder / INFECTION_FILE_PATTERN.format(draw_id=draw_id)

    def get_covariate_file(self, covariate_name):
        return self.covariate_dir / f'{covariate_name}.csv'


class VersionAlreadyExists(RuntimeError):
    pass


class VersionDoesNotExist(RuntimeError):
    pass


def _get_regression_settings_file(regression_version):
    return REGRESSION_OUTPUT / str(regression_version) / 'settings.json'


def _get_forecast_settings_file(forecast_version):
    return FORECAST_OUTPUT / str(forecast_version) / 'settings.json'


def _available_version(version):
    if os.path.exists(version):
        raise VersionAlreadyExists


def _confirm_version_exists(version):
    if not os.path.exists(version):
        raise VersionDoesNotExist


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


@dataclass
class Version:
    """
    - `version_name (str)`: the name of the output version
    """

    version_name: str


@dataclass
class RegressionVersion(Version):
    """
    - `infection_version (str)`: the version of the infection inputs
    - `covariate_version (str)`: the version of the covariate inputs
    - `n_draws (int)`: number of draws
    - `covariates (Dict[str: Dict]): elements of the inner dict:
        - "use_re": (bool)
        - "gprior": (np.array)
        - "bounds": (np.array)
        - "re_var": (float)
    """

    infection_version: str
    covariate_version: str
    n_draws: int
    location_set_version_id: int

    # Spline Arguments
    degree: int
    knots: np.array
    day_shift: int

    # Regression Arguments
    covariates: Dict[str, Dict[str, Union[bool, List, float]]]

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


@dataclass
class ForecastVersion(Version):
    """
    - `regression_version (str)`: the regression version to read from
    - `covariate_scenario_version (str)`: the covariate scenario version to pull
    - `initial_conditions (Dict[str: float])`: initial conditions for the ODE;
        requires keys 'S', 'E', 'I1', '12', and 'R'
    - `solver_dt (float)`: step size for the ODE solver
    """

    regression_version: str
    # covariate_scenario_ids: List[int]

    # Optimization Arguments
    alpha: List[float] = field(default_factory=[0.95, 0.95])
    sigma: List[float] = field(default_factory=[0.20, 0.20])
    gamma1: List[float] = field(default_factory=[0.50, 0.50])
    gamma2: List[float] = field(default_factory=[0.50, 0.50])
    solver_dt: float = field(default=0.1)

    def __post_init__(self):
        pass

    def _settings_to_json(self):
        settings = asdict(self)
        _confirm_version_exists(FORECAST_OUTPUT / self.version_name)
        file = _get_regression_settings_file(self.version_name)
        with open(file, "w") as f:
            json.dump(settings, f)

    def create_version(self):
        _available_version(FORECAST_OUTPUT / self.version_name)
        os.makedirs(FORECAST_OUTPUT / self.version_name)
        self._settings_to_json()

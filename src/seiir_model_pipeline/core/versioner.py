from dataclasses import dataclass, asdict, field
from typing import List, Dict, Union
import os
import json
import numpy as np

from seiir_model_pipeline.core.file_master import _get_regression_settings_file
from seiir_model_pipeline.core.file_master import _get_forecast_settings_file
from seiir_model_pipeline.core.file_master import REGRESSION_OUTPUT, FORECAST_OUTPUT


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
    covariates: Dict[Dict[str, Union[bool, np.ndarray, float]]]

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
        self._settings_to_json()

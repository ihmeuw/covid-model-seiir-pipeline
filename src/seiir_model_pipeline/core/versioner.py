from dataclasses import dataclass, asdict, field
from typing import List, Dict, Union
import os
import json
import numpy as np

from seiir_model_pipeline.core.file_master import Directories, OUTPUT_DIR


class VersionAlreadyExists(RuntimeError):
    pass


class VersionDoesNotExist(RuntimeError):
    pass


def _available_output_version(output_version):
    if os.path.exists(OUTPUT_DIR / output_version):
        raise VersionAlreadyExists


def _check_output_version_exists(output_version):
    if not os.path.exists(OUTPUT_DIR / output_version):
        raise VersionDoesNotExist


def load_settings(directories):
    with open(directories.settings_file) as f:
        settings = json.load(f)
    return settings


@dataclass
class ModelVersion:
    """

    - `version_name (str)`: the name of the output version
    - `infection_version (str)`: the version of the infection inputs
    - `covariate_version (str)`: the version of the covariate inputs
    - `covariates (List[str])`: list of covariate names to use in regression
    - `initial_conditions (Dict[str: float])`: initial conditions for the ODE;
        requires keys 'S', 'E', 'I1', '12', and 'R'
    - `solver_dt (float)`: step size for the ODE solver

    - `covariates (Dict[str: Dict]):
        - Elements of the inner dict:
            - "use_re": (bool)
            - "gprior": (np.array)
            - "bounds": (np.array)
            - "re_var": (float)
    """

    version_name: str

    infection_version: str
    covariate_version: str

    # Spline Arguments
    degree: int
    knots: np.array
    day_shift: int

    # Regression Arguments
    covariates: Dict[Dict[str, Union[bool, np.ndarray, float]]]

    # Optimization Arguments
    alpha: List[float] = field(default_factory=[0.95, 0.95])
    sigma: List[float] = field(default_factory=[0.20, 0.20])
    gamma1: List[float] = field(default_factory=[0.50, 0.50])
    gamma2: List[float] = field(default_factory=[0.50, 0.50])
    solver_dt: float = field(default=0.1)

    def create_version(self):
        directories = Directories(
            infection_version=self.infection_version,
            covariate_version=self.covariate_version,
            output_version=self.version_name
        )
        _available_output_version(self.version_name)
        self._settings_to_json(directories)

    def _settings_to_json(self, directories):
        settings = asdict(self)
        with open(directories.settings_file, "w") as f:
            json.dump(settings, f)

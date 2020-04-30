from dataclasses import dataclass, asdict, field
from typing import List, Dict
import os
import json

from seiir_model_pipeline.core.file_master import Directories, OUTPUT_DIR
from seiir_model.utils import SEIIR_COMPARTMENTS


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
    """

    version_name: str

    infection_version: str
    covariate_version: str

    covariates: List[str]

    initial_conditions: Dict[str: float]
    solver_dt: float = field(default=0.1)

    directories: Directories = field(init=False)

    def __post_init__(self):
        for key in SEIIR_COMPARTMENTS:
            assert key in self.initial_conditions

        self.directories = Directories(
            infection_version=self.infection_version,
            covariate_version=self.covariate_version,
            output_version=self.version_name
        )

    def create_version(self):
        _available_output_version(self.version_name)
        self._settings_to_json()

    def _settings_to_json(self):
        settings = asdict(self)
        with open(self.directories.settings_file, "w") as f:
            json.dump(settings, f)

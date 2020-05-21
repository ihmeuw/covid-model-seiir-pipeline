from dataclasses import dataclass, field
from pathlib import Path
from typing import List, ClassVar

from parse import parse

from covid_shared.shell_tools import mkdir


@dataclass
class ODEPaths:
    # instance attributes
    ode_dir: Path
    read_only: bool = field(default=True)

    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)
    draw_beta_fit_file: ClassVar[str] = 'fit_draw_{draw_id}.csv'
    draw_beta_param_file: ClassVar[str] = 'params_draw_{draw_id}.csv'

    @property
    def beta_fit_dir(self) -> Path:
        return self.ode_dir / 'betas'

    def get_draw_beta_fit_file(self, location_id: int, draw_id: int) -> Path:
        file = self.draw_beta_fit_file.format(draw_id=draw_id)
        return self.beta_fit_dir / str(location_id) / file

    @property
    def parameters_dir(self) -> Path:
        return self.ode_dir / 'parameters'

    def get_draw_beta_param_file(self, draw_id: int) -> Path:
        return self.parameters_dir / self.draw_beta_param_file.format(draw_id=draw_id)

    @property
    def diagnostic_dir(self) -> Path:
        return self.ode_dir / 'diagnostics'

    def make_dirs(self, location_ids: List[int]) -> None:
        if self.read_only:
            raise RuntimeError("tried to create directory structure when ODEPaths was in "
                               "read_only mode. Try instantiating with read_only=False")
        for directory in [self.input_dir, self.beta_fit_dir, self.parameters_dir,
                          self.diagnostic_dir]:
            mkdir(directory, parents=True, exists_ok=True)

        for location_id in location_ids:
            loc_dir = self.beta_fit_dir / str(location_id)
            mkdir(loc_dir, exists_ok=True)


@dataclass
class RegressionPaths:
    # instance attributes
    regression_dir: Path
    read_only: bool = field(default=True)

    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)
    draw_coefficient_file: ClassVar[str] = 'coefficients_{draw_id}.csv'
    draw_beta_fit_file: ClassVar[str] = 'regression_draw_{draw_id}.csv'

    @property
    def beta_fit_dir(self) -> Path:
        return self.regression_dir / 'betas'

    def get_draw_beta_fit_file(self, location_id: int, draw_id: int) -> Path:
        file = self.draw_beta_fit_file.format(draw_id=draw_id)
        return self.beta_fit_dir / str(location_id) / file

    @property
    def coefficient_dir(self) -> Path:
        return self.regression_dir / 'coefficients'

    def get_draw_coefficient_file(self, draw_id: int) -> Path:
        return self.coefficient_dir / self.draw_coefficient_file.format(draw_id=draw_id)

    @property
    def diagnostic_dir(self) -> Path:
        return self.regression_dir / 'diagnostics'

    @property
    def input_dir(self) -> Path:
        return self.regression_dir / 'inputs'

    @property
    def covariate_dir(self) -> Path:
        return self.regression_dir / 'inputs' / 'covariates'

    def make_dirs(self, location_ids: List[int]) -> None:
        if self.read_only:
            raise RuntimeError("tried to create directory structure when RegressionPaths was "
                               "in read_only mode. Try instantiating with read_only=False")
        for directory in [self.beta_fit_dir, self.coefficient_dir, self.diagnostic_dir,
                          self.input_dir, self.covariate_dir]:
            mkdir(directory, parents=True, exists_ok=True)

        for location_id in location_ids:
            loc_dir = self.beta_fit_dir / str(location_id)
            mkdir(loc_dir, exists_ok=True)


@dataclass
class InfectionPaths:
    # instance attributes
    infection_dir: Path

    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)
    infection_file: ClassVar[str] = 'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv'

    def get_location_dir(self, location_id: int):
        matches = [m for m in self.infection_dir.glob(f"*_{location_id}")]
        num_matches = len(matches)
        if num_matches > 1:
            raise RuntimeError("There is more than one location-specific folder for "
                               f"{location_id}.")
        elif num_matches == 0:
            raise FileNotFoundError("There is not a location-specific folder for "
                                    f"{location_id}.")
        else:
            folder = matches[0]
        return self.infection_dir / folder

    def get_infection_file(self, location_id: int, draw_id: int) -> Path:
        # folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        f = (self.get_location_dir(location_id) / self.infection_file.format(draw_id=draw_id))
        return f


@dataclass
class CovariatePaths:
    covariate_dir: Path

    scenario_file: ClassVar[str] = '{covariate}_{scenario}.csv'

    def get_scenarios_set(self, covariate: str) -> List[str]:
        file_glob = self.scenario_file.format(covariate=covariate, scenario="*")
        matched_files = [m for m in self.covariate_dir.glob(file_glob)]
        parsed_matches = [
            parse(self.scenario_file, matched_file)["scenario"]
            for matched_file in matched_files
        ]
        return parsed_matches

    def get_scenario_file(self, covariate: str, scenario: str):
        return self.

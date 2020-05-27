import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, ClassVar, Dict

from parse import parse

from covid_shared.shell_tools import mkdir
from loguru import logger


@dataclass
class Paths:
    """Shared interface for concrete paths objects that represent
    local directory structures.

    """
    root_dir: Path
    read_only: bool = field(default=True)

    @property
    @abc.abstractmethod
    def directories(self) -> List[Path]:
        """Returns all top level sub-directories."""
        raise NotImplementedError

    @property
    def location_specific_directories(self) -> List[Path]:
        """Returns all top level sub-directories that have location-specific
        sub-directories.

        """
        return []

    def make_dirs(self, location_ids: List[int] = None):
        """Builds the local directory structure."""
        if self.read_only:
            raise RuntimeError(f"Tried to create directory structure when "
                               f"{self.__class__.__name__} was in read_only mode. "
                               f"Try instantiating with read_only=False.")

        if self.location_specific_directories and not location_ids:
            raise RuntimeError('No location ids supplied to population location-'
                               f'specific subdirectories for {self.__class__.__name__}.')

        logger.debug(f'Creating sub-directory structure for {self.__class__.__name__} '
                     f'in {self.root_dir}.')
        for directory in self.directories:
            mkdir(directory, parents=True, exists_ok=True)

        for directory in self.location_specific_directories:
            if location_ids is None:
                raise ValueError("location_ids cannot be none when creating directories "
                                 "for location specific outputs of type "
                                 f"{self.__class__.__name__}")
            for location_id in location_ids:
                mkdir(directory / str(location_id), exists_ok=True)


@dataclass
class ForecastPaths(Paths):
    """Local directory structure of a forecasting data root."""
    beta_scaling_file: ClassVar[str] = "{location_id}_beta_scaling.csv"
    component_draw_file: ClassVar[str] = "draw_{draw}.csv"
    beta_resid_plot_file: ClassVar[str] = 'cases_fit_and_beta_residuals_{location_name}.png'
    final_draw_plot_file: ClassVar[str] = 'final_draws_refflog_{location_name}.png'
    trajectories_plot_file: ClassVar[str] = 'trajectories_{location_name}.png'
    cases_file: ClassVar[str] = 'cases_{location_id}.csv'
    deaths_file: ClassVar[str] = 'deaths_{location_id}.csv'
    reff_file: ClassVar[str] = 'reff_{location_id}.csv'

    @property
    def beta_scaling(self) -> Path:
        """Scaling factors used to align past and forecast betas."""
        return self.root_dir / 'beta_scaling'

    def get_beta_scaling_path(self, location_id: int) -> Path:
        """Retrieves a location specific path to beta scaling parameters"""
        return self.beta_scaling / self.beta_scaling_file.format(location_id)

    @property
    def component_draws(self) -> Path:
        """Folders by location with SEIIR components."""
        return self.root_dir / 'component_draws'

    def get_component_draws_path(self, location_id: int, draw: int) -> Path:
        """Get SEIIR components for a particular location and draw."""
        return self.component_draws / str(location_id) / self.component_draw_file.format(draw)

    @property
    def diagnostics(self) -> Path:
        """Plots of SEIIR component draws, final draws, and residuals."""
        return self.root_dir / 'diagnostics'

    def get_residuals_plot(self, location_name: str) -> Path:
        """Path to residual plots by location."""
        return self.diagnostics / self.beta_resid_plot_file.format(location_name)

    def get_final_draw_plots(self, location_name: str) -> Path:
        """Path to final draw plots by location."""
        return self.diagnostics / self.final_draw_plot_file.format(location_name)

    def get_trajectory_plots(self, location_name: str) -> Path:
        """Path to final trajectory plots by location."""
        return self.diagnostics / self.trajectories_plot_file.format(location_name)

    @property
    def output_draws(self) -> Path:
        """Path to the output draws directories."""
        return self.root_dir / 'output_draws'

    def get_output_cases(self, location_id: int) -> Path:
        """Path to output cases file by location."""
        return self.output_draws / self.cases_file.format(location_id)

    def get_output_reff(self, location_id: int) -> Path:
        """Path to output R effective file by location."""
        return self.output_draws / self.reff_file.format(location_id)

    def get_output_deaths(self, location_id: int) -> Path:
        """Path to output deaths file by location"""
        return self.output_draws / self.deaths_file.format(location_id)

    @property
    def directories(self) -> List[Path]:
        """Returns all top level sub-directories."""
        return [self.beta_scaling, self.component_draws, self.diagnostics, self.output_draws]

    @property
    def location_specific_directories(self) -> List[Path]:
        """Returns all top level sub-directories that have location-specific
        sub-directories.

        """
        return [self.component_draws]


@dataclass
class ODEPaths(Paths):
    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)
    draw_beta_fit_file: ClassVar[str] = 'fit_draw_{draw_id}.csv'
    draw_beta_param_file: ClassVar[str] = 'params_draw_{draw_id}.csv'
    draw_date_file: ClassVar[str] = 'dates_draw_{draw_id}.csv'

    @property
    def beta_fit_dir(self) -> Path:
        return self.root_dir / 'betas'

    def get_draw_beta_fit_file(self, location_id: int, draw_id: int) -> Path:
        file = self.draw_beta_fit_file.format(draw_id=draw_id)
        return self.beta_fit_dir / str(location_id) / file

    @property
    def parameters_dir(self) -> Path:
        return self.root_dir / 'parameters'

    def get_draw_beta_param_file(self, draw_id: int) -> Path:
        return self.parameters_dir / self.draw_beta_param_file.format(draw_id=draw_id)

    @property
    def date_dir(self) -> Path:
        return self.root_dir / 'dates'

    def get_draw_date_file(self, draw_id: int) -> Path:
        return self.date_dir / self.draw_date_file.format(draw_id=draw_id)

    @property
    def diagnostic_dir(self) -> Path:
        return self.root_dir / 'diagnostics'

    @property
    def directories(self) -> List[Path]:
        """Returns all top level sub-directories."""
        return [self.beta_fit_dir, self.parameters_dir, self.diagnostic_dir, self.date_dir]

    @property
    def location_specific_directories(self) -> List[Path]:
        """Returns all top level sub-directories that have location-specific
        sub-directories.

        """
        return [self.beta_fit_dir]


@dataclass
class RegressionPaths(Paths):
    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)
    draw_coefficient_file: ClassVar[str] = 'coefficients_{draw_id}.csv'
    draw_beta_regression_file: ClassVar[str] = 'regression_draw_{draw_id}.csv'
    draw_covariates_file: ClassVar[str] = 'covariate_draw_{draw_id}.csv'
    draw_scenarios_file: ClassVar[str] = 'scenarios_draw_{draw_id}.csv'

    @property
    def beta_regression_dir(self) -> Path:
        return self.root_dir / 'betas'

    def get_draw_beta_regression_file(self, location_id: int, draw_id: int) -> Path:
        file = self.draw_beta_regression_file.format(draw_id=draw_id)
        return self.beta_regression_dir / str(location_id) / file

    @property
    def coefficient_dir(self) -> Path:
        return self.root_dir / 'coefficients'

    def get_draw_coefficient_file(self, draw_id: int) -> Path:
        return self.coefficient_dir / self.draw_coefficient_file.format(draw_id=draw_id)

    @property
    def diagnostic_dir(self) -> Path:
        return self.root_dir / 'diagnostics'

    @property
    def input_dir(self) -> Path:
        return self.root_dir / 'inputs'

    @property
    def covariate_dir(self) -> Path:
        return self.root_dir / 'covariates'

    def get_covariates_file(self, draw_id: int) -> Path:
        return self.covariate_dir / self.draw_covariates_file.format(draw_id=draw_id)

    @property
    def scenario_dir(self) -> Path:
        return self.root_dir / 'scenarios'

    def get_scenarios_file(self, location_id: int, draw_id: int):
        file = self.draw_scenarios_file.format(draw_id=draw_id)
        return self.scenario_dir / str(location_id) / file

    @property
    def info_dir(self):
        return self.scenario_dir / 'info'

    @property
    def directories(self) -> List[Path]:
        """Returns all top level sub-directories."""
        return [self.beta_regression_dir, self.coefficient_dir, self.diagnostic_dir,
                self.input_dir, self.covariate_dir, self.scenario_dir, self.info_dir]

    @property
    def location_specific_directories(self) -> List[Path]:
        """Returns all top level sub-directories that have location-specific
        sub-directories.
        """
        return [self.beta_regression_dir, self.scenario_dir]


@dataclass
class InfectionPaths(Paths):
    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)
    infection_file: ClassVar[str] = 'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv'

    def __post_init__(self):
        if not self.read_only:
            raise RuntimeError('Infection outputs should always be read only.')

    @property
    def directories(self) -> List[Path]:
        return []

    def get_location_dir(self, location_id: int):
        matches = [m for m in self.root_dir.glob(f"*_{location_id}")]
        num_matches = len(matches)
        if num_matches > 1:
            raise RuntimeError("There is more than one location-specific folder for "
                               f"{location_id}.")
        elif num_matches == 0:
            raise FileNotFoundError("There is not a location-specific folder for "
                                    f"{location_id}.")
        else:
            folder = matches[0]
        return self.root_dir / folder

    def get_infection_file(self, location_id: int, draw_id: int) -> Path:
        # folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        f = (self.get_location_dir(location_id) / self.infection_file.format(draw_id=draw_id))
        return f


@dataclass
class CovariatePaths(Paths):
    # class attributes are inferred using ClassVar. See pep 557 (Class Variables)

    scenario_file: ClassVar[str] = "{scenario}_scenario.csv"

    def __post_init__(self):
        if not self.read_only:
            raise RuntimeError('Covariate outputs should always be read only.')

    @property
    def directories(self) -> List[Path]:
        return []

    def get_covariate_dir(self, covariate: str) -> Path:
        return self.root_dir / covariate

    def get_covariate_scenario_to_file_mapping(self, covariate: str) -> Dict[str, Path]:
        covariate_dir = self.get_covariate_dir(covariate)
        matches = [m for m in covariate_dir.glob(self.scenario_file.format(scenario="*"))]
        mapping = {}
        for file in matches:
            key = parse(self.scenario_file, str(file.name))["scenario"]
            file = covariate_dir / file
            mapping[key] = file
        return mapping

    def get_info_files(self, covariate: str) -> List[Path]:
        covariate_dir = self.get_covariate_dir(covariate)
        matches = [covariate_dir / m for m in covariate_dir.glob(f"*info.csv")]
        return matches

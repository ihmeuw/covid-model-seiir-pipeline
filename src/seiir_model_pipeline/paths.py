from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ForecastPaths:
    """Local directory structure of a forecasting data root."""
    forecast_dir: Path

    @property
    def beta_scaling(self) -> Path:
        """Scaling factors used to align past and forecast betas."""
        return self.forecast_dir / 'beta_scaling'

    def get_beta_scaling_path(self, location_id: int) -> Path:
        """Retrieves a location specific path to beta scaling parameters"""
        return self.beta_scaling / f"{location_id}_beta_scaling.csv"

    @property
    def component_draws(self) -> Path:
        """Folders by location with SEIIR components."""
        return self.forecast_dir / 'component_draws'

    def get_component_draws_path(self, location_id: int, draw: int) -> Path:
        """Get SEIIR components for a particular location and draw."""
        return self.component_draws / str(location_id) / f'draw_{draw}.csv'

    @property
    def diagnostics(self) -> Path:
        """Plots of SEIIR component draws, final draws, and residuals."""
        return self.forecast_dir / 'diagnostics'

    def get_residuals_plot(self, location_name: str) -> Path:
        """Path to residual plots by location."""
        return self.diagnostics / f'cases_fit_and_beta_residuals_{location_name}.png'

    def get_final_draw_plots(self, location_name: str) -> Path:
        """Path to final draw plots by location."""
        return self.diagnostics / f'final_draws_refflog_{location_name}.png'

    def get_trajectory_plots(self, location_name: str) -> Path:
        """Path to final trajectory plots by location."""
        return self.diagnostics / f'trajectories_{location_name}.png'

    @property
    def output_draws(self) -> Path:
        """Path to the output draws directories."""
        return self.forecast_dir / 'output_draws'

    def get_output_cases(self, location_id: int) -> Path:
        """Path to output cases file by location."""
        return self.output_draws / f'cases_{location_id}.csv'

    def get_output_reff(self, location_id: int) -> Path:
        """Path to output R effective file by location."""
        return self.output_draws / f'reff_{location_id}.csv'

    def get_output_deaths(self, location_id: int) -> Path:
        """Path to output deaths file by location"""
        return self.output_draws / f'deaths_{location_id}.csv'


@dataclass
class ODEPaths:
    ode_dir: Path

    @property
    def input_dir(self) -> Path:
        return self.ode_dir / 'inputs'

    @property
    def beta_fit_dir(self) -> Path:
        return self.ode_dir / 'betas'

    @property
    def parameters_dir(self) -> Path:
        return self.ode_dir / 'parameters'

    @property
    def diagnostic_dir(self) -> Path:
        return self.ode_dir / 'diagnostics'

    def get_draw_beta_fit_file(self, location_id: int, draw_id: int) -> Path:
        return self.ode_dir / str(location_id) / f'fit_draw_{draw_id}.csv'

    def get_draw_beta_param_file(self, draw_id: int) -> Path:
        return self.ode_dir / f'params_draw_{draw_id}.csv'

    def make_dirs(self, location_ids: List[int]) -> None:
        for directory in [self.input_dir, self.beta_fit_dir, self.parameters_dir,
                          self.diagnostic_dir]:
            directory.mkdir(mode=775, parents=True, exist_ok=True)

        for location_id in location_ids:
            loc_dir = self.beta_fit_dir / str(location_id)
            loc_dir.mkdir(mode=775, parents=True, exist_ok=True)


@dataclass
class RegressionPaths:
    regression_dir: Path

    @property
    def beta_fit_dir(self) -> Path:
        return self.ode_dir / 'betas'

    @property
    def coefficient_dir(self) -> Path:
        return self.regression_dir / 'coefficients'

    @property
    def diagnostic_dir(self) -> Path:
        return self.regression_dir / 'diagnostics'

    @property
    def input_dir(self) -> Path:
        return self.regression_dir / 'inputs'

    @property
    def covariate_dir(self) -> Path:
        return self.regression_dir / 'inputs' / 'covariates'

    def get_draw_coefficient_file(self, draw_id: int) -> Path:
        return self.regression_coefficient_dir / f'coefficients_{draw_id}.csv'

    def get_draw_beta_fit_file(self, location_id: int, draw_id: int) -> Path:
        return self.ode_dir / str(location_id) / f'regression_draw_{draw_id}.csv'

    def make_dirs(self, location_ids: List[int]) -> None:
        for directory in [self.regression_beta_fit_dir, self.regression_parameters_dir,
                          self.regression_coefficient_dir, self.regression_diagnostic_dir,
                          self.regression_input_dir, self.regression_covariates_dir]:
            directory.mkdir(mode=775, parents=True, exist_ok=True)

        for location_id in location_ids:
            loc_dir = self.beta_fit_dir / str(location_id)
            loc_dir.mkdir(mode=775, parents=True, exist_ok=True)


@dataclass
class InfectionPaths:
    infection_dir: Path

    def get_location_dir(self, location_id: int):
        matches = self.infection_dir.glob(f"*_{location_id}")
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
        f = (self.infection_dir.get_location_dir(location_id) /
             'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv'.format(draw_id=draw_id))
        return f

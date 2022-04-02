from typing import Dict, List, Iterable, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
)
from covid_model_seiir_pipeline.pipeline.regression import (
    RegressionDataInterface,
    RegressionSpecification,
)
from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
    ForecastSpecification,
    ScenarioSpecification,
)


class ForecastDataInterface:

    def __init__(self,
                 regression_data_interface: RegressionDataInterface,
                 forecast_root: io.ForecastRoot):
        self.regression_data_interface = regression_data_interface
        self.forecast_root = forecast_root

    @classmethod
    def from_specification(cls, specification: ForecastSpecification) -> 'ForecastDataInterface':
        regression_spec = RegressionSpecification.from_version_root(specification.data.seir_regression_version)
        regression_data_interface = RegressionDataInterface.from_specification(regression_spec)

        forecast_root = io.ForecastRoot(specification.data.output_root,
                                        data_format=specification.data.output_format)

        return cls(
            regression_data_interface=regression_data_interface,
            forecast_root=forecast_root,
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.forecast_root, **prefix_args)

    def get_n_draws(self) -> int:
        return self.regression_data_interface.get_n_draws()

    def get_hospital_params(self):
        return self.regression_data_interface.load_specification().hospital_parameters

    ####################
    # Prior Stage Data #
    ####################

    def load_hierarchy(self, name: str) -> pd.DataFrame:
        return self.regression_data_interface.load_hierarchy(name=name)

    def load_population(self, measure: str) -> pd.DataFrame:
        return self.regression_data_interface.load_population(measure=measure)

    def load_reported_epi_data(self) -> pd.DataFrame:
        return self.regression_data_interface.load_reported_epi_data()

    def load_hospital_census_data(self) -> pd.DataFrame:
        return self.regression_data_interface.load_hospital_census_data()

    def load_hospital_bed_capacity(self) -> pd.DataFrame:
        return self.regression_data_interface.load_hospital_bed_capacity()

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        return self.regression_data_interface.load_total_covid_scalars(draw_id=draw_id)

    def load_seroprevalence(self, draw_id: int = None) -> pd.DataFrame:
        return self.regression_data_interface.load_seroprevalence(draw_id=draw_id)

    def load_sensitivity(self, draw_id: int = None) -> pd.DataFrame:
        return self.regression_data_interface.load_sensitivity(draw_id)

    def load_testing_data(self) -> pd.DataFrame:
        return self.regression_data_interface.load_testing_data()

    def load_covariate(self,
                       covariate: str,
                       covariate_version: str = 'reference',
                       with_observed: bool = False) -> pd.DataFrame:
        return self.regression_data_interface.load_covariate(covariate, covariate_version, with_observed)

    def load_covariates(self, covariates: Iterable[str]) -> pd.DataFrame:
        return self.regression_data_interface.load_covariates(covariates)

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        return self.regression_data_interface.load_covariate_info(covariate, info_type)

    def load_mandate_data(self, mobility_scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        percent_mandates = self.load_covariate_info('mobility', f'percent_mandates_{mobility_scenario}')
        mandate_effects = self.load_covariate_info('mobility', 'effect')
        return percent_mandates, mandate_effects

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return self.regression_data_interface.load_variant_prevalence(scenario)

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return self.regression_data_interface.load_waning_parameters(measure)

    def load_vaccine_summary(self, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_vaccine_summary(columns=columns)

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return self.regression_data_interface.load_vaccine_uptake(scenario)

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return self.regression_data_interface.load_vaccine_risk_reduction(scenario)

    def load_covariate_options(self, draw_id: int = None) -> Dict:
        return self.regression_data_interface.load_covariate_options(draw_id)

    def load_fit_ode_params(self, draw_id: int) -> pd.DataFrame:
        return self.regression_data_interface.load_ode_params(draw_id)

    def load_phis(self, draw_id: int) -> pd.DataFrame:
        return self.regression_data_interface.load_phis(draw_id)

    def load_input_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_input_epi_measures(draw_id, columns)

    def load_rates_data(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_rates_data(draw_id, columns)

    def load_rates(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_rates(draw_id, columns)

    def load_posterior_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_posterior_epi_measures(draw_id, columns)

    def load_past_compartments(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_compartments(draw_id, columns)

    def load_fit_beta(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_fit_beta(draw_id, columns)

    def load_final_seroprevalence(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.regression_data_interface.load_final_seroprevalence(draw_id, columns)

    def load_summary(self, measure: str) -> pd.DataFrame:
        return self.regression_data_interface.load_summary(measure)

    def load_location_ids(self) -> List[int]:
        return self.regression_data_interface.load_location_ids()

    def load_regression_beta(self, draw_id: int) -> pd.DataFrame:
        return self.regression_data_interface.load_regression_beta(draw_id=draw_id)

    def load_coefficients(self, draw_id: int) -> pd.DataFrame:
        return self.regression_data_interface.load_coefficients(draw_id=draw_id)

    def load_hospitalizations(self, measure: str) -> pd.DataFrame:
        return self.regression_data_interface.load_hospitalizations(measure)

    #####################
    # Forecast data I/O #
    #####################

    def save_specification(self, specification: ForecastSpecification) -> None:
        io.dump(specification.to_dict(), self.forecast_root.specification())

    def load_specification(self) -> ForecastSpecification:
        spec_dict = io.load(self.forecast_root.specification())
        return ForecastSpecification.from_dict(spec_dict)

    def save_raw_covariates(self, covariates: pd.DataFrame, scenario: str, draw_id: int) -> None:
        io.dump(covariates, self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))

    def load_raw_covariates(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))

    def save_ode_params(self, ode_params: pd.DataFrame, scenario: str, draw_id: int) -> None:
        io.dump(ode_params, self.forecast_root.ode_params(scenario=scenario, draw_id=draw_id))

    def load_ode_params(self, scenario: str, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.forecast_root.ode_params(scenario=scenario, draw_id=draw_id, columns=columns))

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(forecasts, self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))

    def load_components(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))

    def save_beta_scales(self, scales: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(scales, self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def save_beta_residual(self, residual: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(residual, self.forecast_root.beta_residual(scenario=scenario, draw_id=draw_id))

    def load_beta_residual(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_residual(scenario=scenario, draw_id=draw_id))

    def save_raw_outputs(self, raw_outputs: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(raw_outputs, self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    def load_raw_outputs(self, scenario: str, draw_id: int, columns: List[str] = None):
        return io.load(self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id, columns=columns))

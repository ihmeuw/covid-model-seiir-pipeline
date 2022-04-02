from typing import Dict, List

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
)
from covid_model_seiir_pipeline.pipeline.forecasting import (
    ForecastSpecification,
    ForecastDataInterface,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import (
    OOSHoldoutSpecification,
)


class OOSHoldoutDataInterface:

    def __init__(self,
                 forecast_data_interface: ForecastDataInterface,
                 oos_root: io.OOSHoldoutRoot):
        self.forecast_data_interface = forecast_data_interface
        self.oos_root = oos_root

    @classmethod
    def from_specification(cls, specification: OOSHoldoutSpecification) -> 'OOSHoldoutDataInterface':
        forecast_spec = ForecastSpecification.from_version_root(specification.data.seir_forecast_version)
        forecast_data_interface = ForecastDataInterface.from_specification(forecast_spec)
        return cls(
            forecast_data_interface=forecast_data_interface,
            oos_root=io.OOSHoldoutRoot(specification.data.output_root,
                                       data_format=specification.data.output_format),
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.oos_root, **prefix_args)

    def get_n_draws(self):
        return self.forecast_data_interface.get_n_draws()

    ##########################
    # Data from other stages #
    ##########################

    def load_location_ids(self) -> List[int]:
        return self.forecast_data_interface.load_location_ids()

    def load_hierarchy(self, name: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_hierarchy(name=name)

    def load_population(self, measure: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_population(measure=measure)

    def load_fit_beta(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_fit_beta(draw_id, columns)

    def load_covariates(self):
        regression_spec = self.forecast_data_interface.regression_data_interface.load_specification()
        return self.forecast_data_interface.load_covariates(list(regression_spec.covariates))

    def load_priors(self):
        regression_spec = self.forecast_data_interface.regression_data_interface.load_specification()
        return self.forecast_data_interface.regression_data_interface.load_priors(
            regression_spec.covariates.values(),
        )

    def load_prior_run_coefficients(self, draw_id: int):
        return self.forecast_data_interface.load_coefficients(draw_id=draw_id)

    def load_input_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_input_epi_measures(draw_id, columns)

    def load_past_compartments(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_past_compartments(draw_id, columns)

    def load_fit_ode_params(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_fit_ode_params(draw_id=draw_id)

    def load_posterior_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_posterior_epi_measures(draw_id, columns)

    def load_rates(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_rates(draw_id, columns)

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_vaccine_uptake(scenario)

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_vaccine_risk_reduction(scenario)

    def load_phis(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_phis(draw_id)

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_variant_prevalence(scenario)

    def load_hospitalizations(self, measure: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_hospitalizations(measure)

    def get_hospital_params(self):
        return self.forecast_data_interface.get_hospital_params()

    def load_raw_outputs(self, scenario: str, draw_id: int, columns: List[str] = None):
        return self.forecast_data_interface.load_raw_outputs(scenario, draw_id, columns)

    ################
    # OOS data I/O #
    ################

    def save_specification(self, specification: OOSHoldoutSpecification) -> None:
        io.dump(specification.to_dict(), self.oos_root.specification())

    def load_specification(self) -> OOSHoldoutSpecification:
        spec_dict = io.load(self.oos_root.specification())
        return OOSHoldoutSpecification.from_dict(spec_dict)

    def save_coefficients(self, coefficients: pd.DataFrame, draw_id: int) -> None:
        io.dump(coefficients, self.oos_root.coefficients(draw_id=draw_id))

    def load_coefficients(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.oos_root.coefficients(draw_id=draw_id))

    def save_regression_beta(self, betas: pd.DataFrame, draw_id: int) -> None:
        io.dump(betas, self.oos_root.beta(draw_id=draw_id))

    def load_regression_beta(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.oos_root.beta(draw_id=draw_id))

    def save_beta_scales(self, scales: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(scales, self.oos_root.beta_scaling(draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return io.load(self.oos_root.beta_scaling(draw_id=draw_id))

    def save_beta_residual(self, residual: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(residual, self.oos_root.beta_residual(draw_id=draw_id))

    def load_beta_residual(self, scenario: str, draw_id: int):
        return io.load(self.oos_root.beta_residual(draw_id=draw_id))

    def save_raw_oos_outputs(self, data: pd.DataFrame, draw_id: int):
        io.dump(data, self.oos_root.raw_oos_outputs(draw_id=draw_id))

    def load_oos_outputs(self, draw_id: int, columns: List[str] = None):
        return io.load(self.oos_root.raw_oos_outputs(draw_id=draw_id, columns=columns))

    def save_deltas(self, data: pd.DataFrame, draw_id: int):
        io.dump(data, self.oos_root.deltas(draw_id=draw_id))

    def load_deltas(self, draw_id: int, columns: List[str] = None):
        return io.load(self.oos_root.deltas(draw_id=draw_id, columns=columns))

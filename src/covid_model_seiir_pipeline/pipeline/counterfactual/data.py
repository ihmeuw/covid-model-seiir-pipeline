from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.regression import (
    HospitalParameters,
)
from covid_model_seiir_pipeline.pipeline.forecasting import (
    ForecastDataInterface,
    ForecastSpecification,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model import (
    RatioData,
    HospitalMetrics,
    HospitalCorrectionFactors,
    HospitalCensusData,
)
from covid_model_seiir_pipeline.pipeline.counterfactual.specification import (
    CounterfactualSpecification,
)


class CounterfactualDataInterface:

    def __init__(self,
                 input_root: io.CounterfactualInputRoot,
                 forecast_root: io.ForecastRoot,
                 output_root: io.CounterfactualOutputRoot):
        self.input_root = input_root
        self.forecast_root = forecast_root
        self.output_root = output_root

    @classmethod
    def from_specification(cls, specification: CounterfactualSpecification) -> 'CounterfactualDataInterface':
        forecast_spec_path = Path(specification.data.forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
        forecast_spec = ForecastSpecification.from_path(forecast_spec_path)
        forecast_root = io.ForecastRoot(specification.data.forecast_version,
                                        data_format=forecast_spec.data.output_format)
        input_root = io.CounterfactualInputRoot(specification.data.counterfactual_input_version, data_format='parquet')
        output_root = io.CounterfactualOutputRoot(specification.data.output_root,
                                                  data_format=specification.data.output_format)
        return cls(
            input_root=input_root,
            forecast_root=forecast_root,
            output_root=output_root,
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.output_root, **prefix_args)

    ############################
    # Regression paths loaders #
    ############################

    def get_n_draws(self) -> int:
        return self._get_forecast_data_interface().get_n_draws()

    def get_infections_metadata(self):
        return self._get_forecast_data_interface().get_infections_metadata()

    def get_model_inputs_metadata(self):
        return self._get_forecast_data_interface().get_model_inputs_metadata()

    def load_location_ids(self) -> List[int]:
        return self._get_forecast_data_interface().load_location_ids()

    def load_population(self) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_population()

    def load_five_year_population(self) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_five_year_population()

    def load_full_data(self) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_full_data()

    def load_total_deaths(self) -> pd.Series:
        return self._get_forecast_data_interface().load_total_deaths()

    def load_betas(self, draw_id: int):
        return self._get_forecast_data_interface().load_betas(draw_id=draw_id)

    def load_counterfactual_beta(self, scenario: str, draw_id: int):
        beta = io.load(self.input_root.beta(scenario=scenario, columns=[f'draw_{draw_id}']))
        return beta[f'draw_{draw_id}'].rename('beta').dropna()

    def load_covariate(self, covariate: str, covariate_version: str, with_observed: bool = False) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_covariate(
            covariate, covariate_version, with_observed,
        )

    def load_covariates(self, covariates: Dict[str, str]) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_covariates(
            covariates,
        )

    def load_vaccinations(self, vaccine_scenario: str) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_vaccinations(
            vaccine_scenario,
        )

    def load_counterfactual_vaccinations(self, vaccine_scenario: str) -> pd.DataFrame:
        return io.load(self.input_root.vaccine_coverage(vaccine_scenario=vaccine_scenario))

    def load_mobility_info(self, info_type: str) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_mobility_info(
            info_type,
        )

    def load_mandate_data(self, mobility_scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._get_forecast_data_interface().load_mandate_data(
            mobility_scenario,
        )

    def load_variant_prevalence(self, variant_scenario: str) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_variant_prevalence(
            variant_scenario,
        )

    def load_counterfactual_variant_prevalence(self, variant_scenario: str) -> pd.DataFrame:
        return io.load(self.input_root.variant_prevalence(scenario=variant_scenario))

    def load_coefficients(self, draw_id: int) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_coefficients(draw_id=draw_id)

    def load_compartments(self, draw_id: int) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_compartments(draw_id=draw_id)

    def load_ode_parameters(self, draw_id: int) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_ode_parameters(draw_id=draw_id)

    def load_past_infections(self, draw_id: int) -> pd.Series:
        return self._get_forecast_data_interface().load_past_infections(draw_id=draw_id)

    def load_em_scalars(self) -> pd.Series:
        return self._get_forecast_data_interface().load_em_scalars()

    def load_past_deaths(self, draw_id: int) -> pd.Series:
        return self._get_forecast_data_interface().load_past_deaths(draw_id=draw_id)

    def get_hospital_parameters(self) -> HospitalParameters:
        return self._get_forecast_data_interface().get_hospital_parameters()

    def load_hospital_usage(self) -> HospitalMetrics:
        return self._get_forecast_data_interface().load_hospital_usage()

    def load_hospital_correction_factors(self) -> HospitalCorrectionFactors:
        return self._get_forecast_data_interface().load_hospital_correction_factors()

    def load_hospital_census_data(self) -> HospitalCensusData:
        return self._get_forecast_data_interface().load_hospital_census_data()

    def load_ifr(self, draw_id: int) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_ifr(draw_id=draw_id)

    def load_ihr(self, draw_id: int) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_ihr(draw_id=draw_id)

    def load_idr(self, draw_id: int) -> pd.DataFrame:
        return self._get_forecast_data_interface().load_idr(draw_id=draw_id)

    def load_ratio_data(self, draw_id: int) -> RatioData:
        return self._get_forecast_data_interface().load_ratio_data(draw_id=draw_id)

    ##########################
    # Covariate data loaders #
    ##########################

    def check_covariates(self, scenarios: Dict) -> List[str]:
        return self._get_forecast_data_interface().check_covariates(scenarios)

    def get_covariate_version(self, covariate_name: str, scenario: str) -> str:
        return self._get_forecast_data_interface().get_covariate_version(covariate_name, 'reference')

    #####################
    # Forecast data I/O #
    #####################

    def save_specification(self, specification: CounterfactualSpecification) -> None:
        io.dump(specification.to_dict(), self.output_root.specification())

    def load_specification(self) -> CounterfactualSpecification:
        spec_dict = io.load(self.output_root.specification())
        return CounterfactualSpecification.from_dict(spec_dict)

    def load_forecast_specification(self) -> ForecastSpecification:
        spec_dict = io.load(self.forecast_root.specification())
        return ForecastSpecification.from_dict(spec_dict)

    def load_raw_covariates(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))

    def save_ode_params(self, ode_params: pd.DataFrame, scenario: str, draw_id: int) -> None:
        io.dump(ode_params, self.output_root.ode_params(scenario=scenario, draw_id=draw_id))

    def load_ode_params(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.output_root.ode_params(scenario=scenario, draw_id=draw_id))

    def load_forecast_ode_params(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.forecast_root.ode_params(scenario=scenario, draw_id=draw_id))

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(forecasts, self.output_root.component_draws(scenario=scenario, draw_id=draw_id))

    def load_components(self, scenario: str, draw_id: int):
        return io.load(self.output_root.component_draws(scenario=scenario, draw_id=draw_id))

    def load_forecast_components(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def load_beta_residual(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_residual(scenario=scenario, draw_id=draw_id))

    def save_raw_outputs(self, raw_outputs: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(raw_outputs, self.output_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    def load_raw_outputs(self, scenario: str, draw_id: int):
        return io.load(self.output_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    def load_forecast_raw_outputs(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    #########################
    # Non-interface helpers #
    #########################

    def _get_forecast_data_interface(self):
        forecast_spec = ForecastSpecification.from_dict(io.load(self.forecast_root.specification()))
        forecast_di = ForecastDataInterface.from_specification(forecast_spec)
        return forecast_di

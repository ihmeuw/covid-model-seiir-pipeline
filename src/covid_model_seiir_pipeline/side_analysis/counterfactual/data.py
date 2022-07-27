from typing import List

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io
)
from covid_model_seiir_pipeline.pipeline.forecasting import (
    ForecastDataInterface,
    ForecastSpecification,
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.specification import (
    CounterfactualSpecification,
)


class CounterfactualDataInterface:

    def __init__(self,
                 forecast_data_interface: ForecastDataInterface,
                 input_root: io.CounterfactualInputRoot,
                 output_root: io.CounterfactualRoot):
        self.forecast_data_interface = forecast_data_interface
        self.input_root = input_root
        self.output_root = output_root

    def __getattr__(self, item):
        return getattr(self.forecast_data_interface, item)

    @classmethod
    def from_specification(cls, specification: CounterfactualSpecification) -> 'CounterfactualDataInterface':
        forecast_spec = ForecastSpecification.from_version_root(specification.data.seir_forecast_version)
        forecast_data_interface = ForecastDataInterface.from_specification(forecast_spec)
        return cls(
            forecast_data_interface=forecast_data_interface,
            input_root=io.CounterfactualInputRoot(
                specification.data.seir_counterfactual_input_version, data_format='parquet'
            ),
            output_root=io.CounterfactualRoot(
                specification.data.output_root, data_format=specification.data.output_format
            ),
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.output_root, **prefix_args)

    def get_n_draws(self) -> int:
        return self.forecast_data_interface.get_n_draws()

    def get_covariate_version(self, covariate_name: str, scenario: str) -> str:
        specification = self.load_specification()
        counterfactual_version = specification.scenarios[scenario].to_dict().get(covariate_name)

        forecast_spec = self.forecast_data_interface.load_specification()
        forecast_version = forecast_spec.scenarios['reference'].covariates[covariate_name]

        covariate_version = counterfactual_version if counterfactual_version else forecast_version
        return covariate_version

    def load_location_ids(self):
        return self.forecast_data_interface.load_location_ids()

    def load_past_compartments(self, draw_id: int):
        return self.forecast_data_interface.load_past_compartments(draw_id)

    def load_counterfactual_beta(self, scenario: str, draw_id: int):
        if scenario:
            beta = io.load(self.input_root.beta(scenario=scenario, draw_id=draw_id))
        else:
            beta = self.forecast_data_interface.load_raw_outputs(
                scenario='reference', draw_id=draw_id, columns=['beta']
            )['beta']
        return beta

    def load_forecast_ode_params(self, draw_id: int):
        return self.forecast_data_interface.load_ode_params(
            scenario='reference', draw_id=draw_id
        )

    def load_counterfactual_vaccine_uptake(self, scenario: str):
        if scenario:
            uptake = io.load(self.input_root.vaccine_uptake(covariate_scenario=scenario))
        else:
            uptake = self.forecast_data_interface.load_vaccine_uptake('reference')
        return uptake

    def load_counterfactual_vaccine_risk_reduction(self, scenario: str):
        if scenario:
            etas = io.load(self.input_root.etas(covariate_scenario=scenario))
        else:
            etas = self.forecast_data_interface.load_vaccine_risk_reduction('reference')
        return etas

    def load_rates(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_rates(draw_id)

    def load_phis(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_phis(draw_id)

    def load_raw_covariates(self, scenario: str, draw_id: int):
        # TODO: replace reference covariates with counterfactual covariates.
        return self.forecast_data_interface.load_raw_covariates('reference', draw_id)

    def load_beta_residuals(self, draw_id: int, scenario: str):
        # TODO: Think about this
        return self.forecast_data_interface.load_beta_residual(scenario='reference', draw_id=draw_id)

    def load_beta_scales(self, draw_id: int, scenario: str):
        # TODO: Think about this
        return self.forecast_data_interface.load_beta_scales(scenario=scenario, draw_id=draw_id)

    def load_hospitalizations(self, measure: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_hospitalizations(measure)

    def get_hospital_params(self):
        return self.forecast_data_interface.get_hospital_params()

    ###########################
    # Counterfactual data I/O #
    ###########################

    def save_specification(self, specification: CounterfactualSpecification) -> None:
        io.dump(specification.to_dict(), self.output_root.specification())

    def load_specification(self) -> CounterfactualSpecification:
        spec_dict = io.load(self.output_root.specification())
        return CounterfactualSpecification.from_dict(spec_dict)

    def save_ode_params(self, ode_params: pd.DataFrame, scenario: str, draw_id: int) -> None:
        io.dump(ode_params, self.output_root.ode_params(scenario=scenario, draw_id=draw_id))

    def load_ode_params(self, scenario: str, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return io.load(self.output_root.ode_params(scenario=scenario, draw_id=draw_id, columns=columns))

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(forecasts, self.output_root.component_draws(scenario=scenario, draw_id=draw_id))

    def load_components(self, scenario: str, draw_id: int):
        return io.load(self.output_root.component_draws(scenario=scenario, draw_id=draw_id))

    def save_raw_outputs(self, raw_outputs: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(raw_outputs, self.output_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    def load_raw_outputs(self, scenario: str, draw_id: int, columns: List[str] = None):
        return io.load(self.output_root.raw_outputs(scenario=scenario, draw_id=draw_id, columns=columns))

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)

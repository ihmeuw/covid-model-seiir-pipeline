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

    #####################
    # Preprocessed Data #
    #####################

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


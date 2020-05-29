from pathlib import Path
from typing import List

import pandas as pd


from covid_model_seiir_pipeline.paths import (RegressionPaths, ODEPaths, ForecastPaths)
from covid_model_seiir_pipeline.static_vars import (FIT_SPECIFICATION_FILE,
                                                    REGRESSION_SPECIFICATION_FILE,
                                                    COVARIATE_COL_DICT)
from covid_model_seiir_pipeline.forecasting.specification import ForecastData
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification


class ForecastDataInterface:

    def __init__(self, forecast_root: Path, regression_root: Path, ode_fit_root: Path,
                 location_file: Path):

        self.forecast_paths = ForecastPaths(forecast_root)

        # setup regression dirs
        self.regression_paths = RegressionPaths(regression_root)

        # setup ode dirs
        self.ode_paths = ODEPaths(ode_fit_root)

        self.location_metadata_file = location_file

    def load_location_ids(self) -> List[int]:
        return pd.read_csv(self.location_metadata_file)["location_id"].tolist()

    def load_beta_fit(self, draw_id: int, location_id: int) -> pd.DataFrame:
        beta_fit_file = self.ode_paths.get_draw_beta_fit_file(location_id=location_id,
                                                              draw_id=draw_id)
        return pd.read_csv(beta_fit_file)

    def load_beta_params(self, draw_id: int) -> pd.DataFrame:
        beta_params_file = self.ode_paths.get_draw_beta_param_file(draw_id)
        return pd.read_csv(beta_params_file)

    def load_covariate_scenarios(self, draw_id: int, location_id: int,
                                 covariate_scenarios: List[str]) -> pd.DataFrame:
        scenario_file = self.regression_paths.get_scenarios_file(location_id=location_id,
                                                                 draw_id=draw_id)
        df = pd.read_csv(scenario_file)
        index_columns = [COVARIATE_COL_DICT['COL_DATE'], COVARIATE_COL_DICT['COL_LOC_ID']]
        return df[index_columns + covariate_scenarios]

    def load_regression_coefficients(self, draw_id: int) -> pd.DataFrame:
        file = self.regression_paths.get_draw_coefficient_file(draw_id)
        return pd.read_csv(file)

    def save_components(self, df: pd.DataFrame, draw_id: int, location_id: int) -> None:
        file = self.forecast_paths.get_component_draws_path(location_id=location_id,
                                                            draw_id=draw_id)
        return df.to_csv(file, index=False)

    def save_beta_scales(self, scales: List[int], location_id: int):
        df_scales = pd.DataFrame({
            'beta_scales': scales
        })
        file = self.forecast_paths.get_beta_scaling_path(location_id)
        df_scales.to_csv(file, index=False)

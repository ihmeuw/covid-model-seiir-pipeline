from pathlib import Path
from typing import List, Dict

import pandas as pd

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.static_vars import COVARIATE_COL_DICT


class ForecastDataInterface:

    def __init__(self,
                 forecast_paths: paths.ForecastPaths,
                 regression_paths: paths.RegressionPaths,
                 covariate_paths: paths.CovariatePaths):
        self.forecast_paths = forecast_paths
        self.regression_paths = regression_paths
        self.covariate_paths = covariate_paths

    @classmethod
    def from_specification(cls, specification: ForecastSpecification) -> 'ForecastDataInterface':
        forecast_paths = paths.ForecastPaths(specification.data.output_root, read_only=False)
        regression_paths = paths.RegressionPaths(specification.data.regression_version)
        covariate_paths = paths.CovariatePaths(specification.data.covariate_version)
        return cls(
            forecast_paths=forecast_paths,
            regression_paths=regression_paths,
            covariate_paths=covariate_paths
        )

    def make_dirs(self):
        self.forecast_paths.make_dirs()

    def load_location_ids(self) -> List[int]:
        return pd.read_csv(self.location_metadata_file)["location_id"].tolist()

    def get_location_name_from_id(self, location_id: int) -> str:
        df = pd.read_csv(self.location_metadata_file)
        location_name = df.loc[df.location_id == location_id]['location_name'].iloc[0]
        return location_name

    def load_beta_fit(self, draw_id: int, location_id: int) -> pd.DataFrame:
        beta_fit_file = self.ode_paths.get_draw_beta_fit_file(location_id=location_id,
                                                              draw_id=draw_id)
        return pd.read_csv(beta_fit_file)

    def load_beta_params(self, draw_id: int) -> pd.DataFrame:
        beta_params_file = self.ode_paths.get_draw_beta_param_file(draw_id)
        df = pd.read_csv(beta_params_file)
        return df.set_index('params')['values'].to_dict()

    def load_covariate_scenarios(self, draw_id: int, location_id: int,
                                 scenario_covariate_mapping: Dict[str, str]
                                 ) -> pd.DataFrame:
        # import file
        scenario_file = self.regression_paths.get_scenarios_file(location_id=location_id,
                                                                 draw_id=draw_id)
        df = pd.read_csv(scenario_file)

        # rename scenarios
        missing_cols = set(list(scenario_covariate_mapping.keys())) - set(df.columns)
        if missing_cols:
            raise ValueError("One or more scenarios missing from scenario pool. Missing "
                             f"scenarios are: {missing_cols}")
        df = df.rename(columns=scenario_covariate_mapping)

        # subset columns
        index_columns = [COVARIATE_COL_DICT['COL_DATE'], COVARIATE_COL_DICT['COL_LOC_ID']]
        data_columns = list(scenario_covariate_mapping.values())
        df = df[index_columns + data_columns]
        return df

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

    def load_beta_scales(self, location_id: int):
        file = self.forecast_paths.get_beta_scaling_path(location_id)
        return pd.read_csv(file)

    def load_infections(self, location_id: int, draw_id: int) -> pd.DataFrame:
        file = self.infection_paths.get_infection_file(location_id=location_id,
                                                       draw_id=draw_id)
        return pd.read_csv(file)

    def load_component_forecasts(self, location_id: int, draw_id: int):
        file = self.forecast_paths.get_component_draws_path(location_id=location_id,
                                                            draw_id=draw_id)
        return pd.read_csv(file)

    def save_cases(self, df: pd.DataFrame, location_id: int):
        file = self.forecast_paths.get_output_cases(location_id=location_id)
        df.to_csv(file, index=False)

    def save_deaths(self, df: pd.DataFrame, location_id: int):
        file = self.forecast_paths.get_output_deaths(location_id=location_id)
        df.to_csv(file, index=False)

    def save_reff(self, df: pd.DataFrame, location_id: int):
        file = self.forecast_paths.get_output_reff(location_id=location_id)
        df.to_csv(file, index=False)

    def load_cases(self, location_id: int) -> pd.DataFrame:
        file = self.forecast_paths.get_output_cases(location_id=location_id)
        return pd.read_csv(file)

    def load_deaths(self, location_id: int) -> pd.DataFrame:
        file = self.forecast_paths.get_output_deaths(location_id=location_id)
        return pd.read_csv(file)

    def load_reff(self, location_id: int) -> pd.DataFrame:
        file = self.forecast_paths.get_output_reff(location_id=location_id)
        return pd.read_csv(file)

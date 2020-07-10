from typing import List, Dict

from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification, ScenarioSpecification
from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
    Keys as MKeys,
)
from covid_model_seiir_pipeline.static_vars import COVARIATE_COL_DICT


class ForecastDataInterface:

    def __init__(self,
                 forecast_paths: paths.ForecastPaths,
                 regression_paths: paths.RegressionPaths,
                 covariate_paths: paths.CovariatePaths,
                 regression_marshall,
                 forecast_marshall):
        self.forecast_paths = forecast_paths
        self.regression_paths = regression_paths
        self.covariate_paths = covariate_paths
        self.regression_marshall = regression_marshall
        self.forecast_marshall = forecast_marshall

    @classmethod
    def from_specification(cls, specification: ForecastSpecification) -> 'ForecastDataInterface':
        forecast_paths = paths.ForecastPaths(specification.data.output_root,
                                             read_only=False,
                                             scenarios=list(specification.scenarios))
        regression_paths = paths.RegressionPaths(specification.data.regression_version)
        covariate_paths = paths.CovariatePaths(specification.data.covariate_version)
        regression_marshall = CSVMarshall.from_paths(regression_paths)
        forecast_marshall = CSVMarshall.from_paths(forecast_paths)
        return cls(
            forecast_paths=forecast_paths,
            regression_paths=regression_paths,
            covariate_paths=covariate_paths,
            regression_marshall=regression_marshall,
            forecast_marshall=forecast_marshall,
        )

    def make_dirs(self):
        self.forecast_paths.make_dirs()

    def get_n_draws(self) -> int:
        # Fixme: Gross
        with self.regression_paths.regression_specification.open() as regression_spec_file:
            regression_spec = yaml.full_load(regression_spec_file)
        return regression_spec['parameters']['n_draws']

    def load_location_ids(self) -> List[int]:
        with self.regression_paths.location_metadata.open() as location_file:
            location_ids = yaml.full_load(location_file)
        return location_ids

    def check_covariates(self, scenarios: Dict[str, ScenarioSpecification]):
        with self.regression_paths.regression_specification.open() as regression_spec_file:
            regression_spec = yaml.load(regression_spec_file)
        forecast_version = str(self.covariate_paths.root_dir)
        regression_version = regression_spec['data']['covariate_version']
        if not forecast_version == regression_version:
            logger.warning(f'Forecast covariate version {forecast_version} does not match '
                           f'regression covariate version {regression_version}. If the two covariate'
                           f'versions have different data in the past, the regression coefficients '
                           f'used for prediction may not be valid.')

        regression_covariates = set(regression_spec['covariates'])

        for name, scenario in scenarios.items():
            if not set(scenario.covariates) == regression_covariates:
                raise ValueError('Forecast covariates must match the covariates used in regression.\n'
                                 f'Forecast covariates:   {sorted(list(scenario.covariates))}.\n'
                                 f'Regression covariates: {sorted(list(regression_covariates))}.')

            for covariate, covariate_version in scenario.covariates:
                data_file = self.covariate_paths.get_covariate_scenario_file(covariate, covariate_version)
                if not data_file.exists():
                    raise FileNotFoundError(f'No {covariate_version} file found for covariate {covariate}.')

    def load_beta_fit(self, draw_id: int, location_id: int) -> pd.DataFrame:
        beta_fit_file = self.ode_paths.get_draw_beta_fit_file(location_id=location_id,
                                                              draw_id=draw_id)
        return pd.read_csv(beta_fit_file)

    def load_beta_params(self, draw_id: int) -> pd.DataFrame:
        df = self.regression_marshall.load(key=MKeys.parameter(draw_id=draw_id))
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
        return self.regression_marshall.load(MKeys.coefficient(draw_id))

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

    # new methods to be used after forecasting is fully fixed to be by draw_id
    def save_components_futurerefactor(self, df: pd.DataFrame, scenario: str, draw_id: int):
        self.forecast_marshall.dump(df, key=MKeys.components(scenario=scenario, draw_id=draw_id))

    def load_component_forecasts_futurerefactor(self, scenario: str, draw_id: int):  # TODO: load_components?
        return self.forecast_marshall.load(MKeys.components(scenario=scenario, draw_id=draw_id))

    def save_beta_scales_futurerefactor(self, df: pd.DataFrame, scenario: str, draw_id: int):
        # TODO: this does not match save_beta_scales at all
        #       but James pointed Mike to fixture data at
        #       /ihme/covid-19/seir-pipeline-outputs/forecast/2020_07_08.integration_test_2/
        #       and so it is expected that the "beta_scales" argument will
        #       resemble the data in the "beta_scaling" subdirectory
        self.forecast_marshall.dump(df, key=MKeys.beta_scales(scenario=scenario, draw_id=draw_id))

    def load_beta_scales_futurerefactor(self, scenario: str, draw_id: int):
        return self.forecast_marshall.load(MKeys.beta_scales(scenario=scenario, draw_id=draw_id))

from functools import reduce
from pathlib import Path
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


# TODO: move data interfaces up a package level and fuse with regression data interface.

class ForecastDataInterface:

    def __init__(self,
                 forecast_paths: paths.ForecastPaths,
                 regression_paths: paths.RegressionPaths,
                 covariate_paths: paths.CovariatePaths,
                 regression_marshall,
                 forecast_marshall):
        # TODO: only hang on to marshalls here.
        self.forecast_paths = forecast_paths
        self.regression_paths = regression_paths
        self.covariate_paths = covariate_paths
        self.regression_marshall = regression_marshall
        self.forecast_marshall = forecast_marshall

    @classmethod
    def from_specification(cls, specification: ForecastSpecification) -> 'ForecastDataInterface':
        forecast_paths = paths.ForecastPaths(Path(specification.data.output_root),
                                             read_only=False,
                                             scenarios=list(specification.scenarios))
        regression_paths = paths.RegressionPaths(Path(specification.data.regression_version))
        covariate_paths = paths.CovariatePaths(Path(specification.data.covariate_version))
        # TODO: specification of marshall type from inference on inputs and
        #   configuration on outputs.
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
            regression_spec = yaml.full_load(regression_spec_file)
        forecast_version = str(self.covariate_paths.root_dir)
        regression_version = regression_spec['data']['covariate_version']
        if not forecast_version == regression_version:
            logger.warning(f'Forecast covariate version {forecast_version} does not match '
                           f'regression covariate version {regression_version}. If the two covariate'
                           f'versions have different data in the past, the regression coefficients '
                           f'used for prediction may not be valid.')

        regression_covariates = set(regression_spec['covariates'])

        for name, scenario in scenarios.items():
            if set(scenario.covariates).symmetric_difference(regression_covariates) > {'intercept'}:
                raise ValueError('Forecast covariates must match the covariates used in regression.\n'
                                 f'Forecast covariates:   {sorted(list(scenario.covariates))}.\n'
                                 f'Regression covariates: {sorted(list(regression_covariates))}.')

            if 'intercept' in scenario.covariates:
                # Shouldn't really be specified, but might be copied over from
                # regression.  No harm really in just deleting it.
                del scenario.covariates['intercept']

            for covariate, covariate_version in scenario.covariates.items():
                data_file = self.covariate_paths.get_covariate_scenario_file(covariate, covariate_version)
                if not data_file.exists():
                    raise FileNotFoundError(f'No {covariate_version} file found for covariate {covariate}.')

    def load_regression_coefficients(self, draw_id: int) -> pd.DataFrame:
        return self.regression_marshall.load(MKeys.coefficient(draw_id))

    # TODO: inverse is RegressionDataInterface.save_date_file
    def load_dates_df(self, draw_id: int) -> pd.DataFrame:
        dates_df = self.regression_marshall.load(key=MKeys.date(draw_id))
        dates_df['start_date'] = pd.to_datetime(dates_df['start_date'])
        dates_df['end_date'] = pd.to_datetime(dates_df['end_date'])
        return dates_df

    # TODO: inverse is RegressionDataInterface.save_regression_betas
    def load_beta_regression(self, draw_id: int) -> pd.DataFrame:
        beta_regression = self.regression_marshall.load(key=MKeys.regression_beta(draw_id))
        beta_regression['date'] = pd.to_datetime(beta_regression['date'])
        return beta_regression

    # TODO: inverse is RegressionDataInterface.save_location_data
    def load_infection_data(self, draw_id: int) -> pd.DataFrame:
        infection_data = self.regression_marshall.load(key=MKeys.location_data(draw_id))
        infection_data['date'] = pd.to_datetime(infection_data['date'])
        return infection_data

    # FIXME: Duplicate code.
    def load_covariate(self, covariate: str, covariate_version: str, location_ids: List[int]) -> pd.DataFrame:
        covariate_path = self.covariate_paths.get_covariate_scenario_file(covariate, covariate_version)
        covariate_df = pd.read_csv(covariate_path)
        index_columns = ['location_id']
        covariate_df = covariate_df.loc[covariate_df['location_id'].isin(location_ids), :]
        if 'date' in covariate_df.columns:
            covariate_df['date'] = pd.to_datetime(covariate_df['date'])
            index_columns.append('date')
        covariate_df = covariate_df.rename(columns={f'{covariate}_{covariate_version}': covariate})
        return covariate_df.loc[:, index_columns + [covariate]].set_index(index_columns)

    def load_covariates(self, scenario: ScenarioSpecification, location_ids: List[int]) -> pd.DataFrame:
        covariate_data = []
        for covariate, covariate_version in scenario.covariates.items():
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate, covariate_version, location_ids))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data.reset_index()

    def load_beta_params(self, draw_id: int) -> Dict[str, float]:
        df = self.regression_marshall.load(key=MKeys.parameter(draw_id=draw_id))
        return df.set_index('params')['values'].to_dict()

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int):
        self.forecast_marshall.dump(forecasts, key=MKeys.components(scenario=scenario, draw_id=draw_id))

    def load_components(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return self.forecast_marshall.load(key=MKeys.components(scenario=scenario, draw_id=draw_id))

    def save_beta_scales(self, scales: pd.DataFrame, scenario: str, draw_id: int):
        self.forecast_marshall.dump(scales, key=MKeys.beta_scales(scenario=scenario, draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return self.forecast_marshall.load(MKeys.beta_scales(scenario=scenario, draw_id=draw_id))

    def save_outputs(self, outputs: pd.DataFrame, scenario: str, draw_id: int):
        self.forecast_marshall.dump(outputs, key=MKeys.forecast_outputs(scenario=scenario, draw_id=draw_id))

    def load_outputs(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return self.forecast_marshall.load(key=MKeys.forecast_outputs(scenario=scenario, draw_id=draw_id))

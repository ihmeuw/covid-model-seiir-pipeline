from functools import reduce
from pathlib import Path
from typing import List, Dict, Union

from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification, ScenarioSpecification, PostprocessingSpecification
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
                 postprocessing_paths: paths.PostprocessingPaths,
                 regression_marshall,
                 forecast_marshall,
                 postprocessing_marshall):
        # TODO: only hang on to marshalls here.
        self.forecast_paths = forecast_paths
        self.regression_paths = regression_paths
        self.covariate_paths = covariate_paths
        self.postprocessing_paths = postprocessing_paths
        self.regression_marshall = regression_marshall
        self.forecast_marshall = forecast_marshall
        self.postprocessing_marshall = postprocessing_marshall

    @classmethod
    def from_specification(cls,
                           specification: ForecastSpecification,
                           postprocessing_specification: PostprocessingSpecification = None) -> 'ForecastDataInterface':
        forecast_paths = paths.ForecastPaths(Path(specification.data.output_root),
                                             read_only=False,
                                             scenarios=list(specification.scenarios))
        regression_paths = paths.RegressionPaths(Path(specification.data.regression_version))
        covariate_paths = paths.CovariatePaths(Path(specification.data.covariate_version))
        # TODO: specification of marshall type from inference on inputs and
        #   configuration on outputs.
        regression_marshall = CSVMarshall.from_paths(regression_paths)
        forecast_marshall = CSVMarshall.from_paths(forecast_paths)

        if postprocessing_specification is not None:
            postprocessing_paths = paths.PostprocessingPaths(
                Path(postprocessing_specification.data.output_root),
                read_only=False,
                scenarios=list(postprocessing_specification.data.include_scenarios)
            )
            postprocessing_marshall = CSVMarshall.from_paths(postprocessing_paths)
        else:
            postprocessing_paths = None
            postprocessing_marshall = None

        return cls(
            forecast_paths=forecast_paths,
            regression_paths=regression_paths,
            covariate_paths=covariate_paths,
            postprocessing_paths=postprocessing_paths,
            regression_marshall=regression_marshall,
            forecast_marshall=forecast_marshall,
            postprocessing_marshall=postprocessing_marshall
        )

    def make_dirs(self):
        self.forecast_paths.make_dirs()
        if self.postprocessing_paths:
            self.postprocessing_paths.make_dirs()

    def get_n_draws(self) -> int:
        # Fixme: Gross
        with self.regression_paths.regression_specification.open() as regression_spec_file:
            regression_spec = yaml.full_load(regression_spec_file)
        return regression_spec['parameters']['n_draws']

    def load_location_ids(self) -> List[int]:
        with self.regression_paths.location_metadata.open() as location_file:
            location_ids = yaml.full_load(location_file)
        return location_ids

    def load_thetas(self, theta_specification: Union[str, int]) -> pd.Series:
        if isinstance(theta_specification, str):
            thetas = pd.read_csv(theta_specification).set_index('location_id')['theta']
        else:
            location_ids = self.load_location_ids()
            thetas = pd.Series(theta_specification,
                               index=pd.Index(location_ids, name='location_id'),
                               name='theta')
        return thetas

    def check_covariates(self, scenarios: Dict[str, ScenarioSpecification]) -> List[str]:
        with self.regression_paths.regression_specification.open() as regression_spec_file:
            regression_spec = yaml.full_load(regression_spec_file)
        forecast_version = str(self.covariate_paths.root_dir)
        regression_version = regression_spec['data']['covariate_version']
        if not forecast_version == regression_version:
            logger.warning(f'Forecast covariate version {forecast_version} does not match '
                           f'regression covariate version {regression_version}. If the two covariate '
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
        return list(regression_covariates)

    def get_infectionator_metadata(self):
        with self.regression_paths.regression_specification.open() as regression_spec_file:
            regression_spec = yaml.full_load(regression_spec_file)
        infectionator_path = Path(regression_spec['data']['infection_version'])
        with (infectionator_path / 'metadata.yaml').open() as metadata_file:
            metadata = yaml.full_load(metadata_file)
        return metadata

    def load_full_data(self):
        metadata = self.get_infectionator_metadata()
        # TODO: metadata abstraction?
        model_inputs_version = metadata['death']['metadata']['model_inputs_metadata']['output_path']
        full_data_path = Path(model_inputs_version) / 'full_data.csv'
        full_data = pd.read_csv(full_data_path)
        full_data['date'] = pd.to_datetime(full_data['Date'])
        full_data = full_data.drop(columns=['Date'])
        return full_data

    def load_elastispliner_inputs(self):
        metadata = self.get_infectionator_metadata()
        deaths_version = Path(metadata['death']['metadata']['output_path'])
        es_inputs = pd.read_csv(deaths_version / 'model_data.csv')
        es_inputs['date'] = pd.to_datetime(es_inputs['date'])
        return es_inputs

    def load_elastispliner_outputs(self):
        metadata = self.get_infectionator_metadata()
        deaths_version = Path(metadata['death']['metadata']['output_path'])
        noisy_outputs = pd.read_csv(deaths_version / 'model_results.csv')
        noisy_outputs['date'] = pd.to_datetime(noisy_outputs['date'])
        smoothed_outputs = pd.read_csv(deaths_version / 'model_results_refit.csv')
        smoothed_outputs['date'] = pd.to_datetime(smoothed_outputs['date'])
        return noisy_outputs, smoothed_outputs

    def load_total_deaths(self):
        """Load cumulative deaths by location."""
        full_data = self.load_full_data()
        total_deaths = full_data.groupby('location_id')['Deaths'].max().rename('deaths').reset_index()
        total_deaths['location_id'] = total_deaths['location_id'].astype(int)
        return total_deaths

    def load_regression_coefficients(self, draw_id: int) -> pd.DataFrame:
        return self.regression_marshall.load(MKeys.coefficient(draw_id))

    # TODO: inverse is RegressionDataInterface.save_date_file
    def load_transition_date(self, draw_id: int) -> pd.Series:
        dates_df = self.regression_marshall.load(key=MKeys.date(draw_id))
        dates_df['end_date'] = pd.to_datetime(dates_df['end_date'])
        transition_date = dates_df.set_index('location_id').sort_index()['end_date'].rename('date')
        return transition_date

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

    def load_covariate(self, covariate: str, covariate_version: str, location_ids: List[int],
                       with_observed: bool = False) -> pd.DataFrame:
        covariate_path = self.covariate_paths.get_covariate_scenario_file(covariate, covariate_version)
        covariate_df = pd.read_csv(covariate_path)
        index_columns = ['location_id']
        if with_observed:
            index_columns.append('observed')
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

    def load_covariate_info(self, covariate: str, info_type: str, location_ids: List[int]):
        info_path = self.covariate_paths.get_covariate_info_file(covariate, info_type)
        info_df = pd.read_csv(info_path)
        index_columns = ['location_id']
        info_df = info_df.loc[info_df['location_id'].isin(location_ids), :]
        if 'date' in info_df.columns:
            info_df['date'] = pd.to_datetime(info_df['date'])
            index_columns.append('date')
        return info_df.set_index(index_columns)

    def save_raw_covariates(self, covariates: pd.DataFrame, scenario: str, draw_id: int, strict: bool):
        self.forecast_marshall.dump(covariates,
                                    key=MKeys.forecast_raw_covariates(scenario=scenario, draw_id=draw_id),
                                    strict=strict)

    def load_raw_covariates(self, scenario: str, draw_id: int):
        covariates = self.forecast_marshall.load(key=MKeys.forecast_raw_covariates(scenario=scenario, draw_id=draw_id))
        covariates['date'] = pd.to_datetime(covariates['date'])
        return covariates

    def load_beta_params(self, draw_id: int) -> Dict[str, float]:
        df = self.regression_marshall.load(key=MKeys.parameter(draw_id=draw_id))
        return df.set_index('params')['values'].to_dict()

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int, strict: bool):
        self.forecast_marshall.dump(forecasts, key=MKeys.components(scenario=scenario, draw_id=draw_id), strict=strict)

    def load_components(self, scenario: str, draw_id: int) -> pd.DataFrame:
        components = self.forecast_marshall.load(key=MKeys.components(scenario=scenario, draw_id=draw_id))
        components['date'] = pd.to_datetime(components['date'])
        return components.set_index(['location_id', 'date'])

    def save_beta_scales(self, scales: pd.DataFrame, scenario: str, draw_id: int):
        self.forecast_marshall.dump(scales, key=MKeys.beta_scales(scenario=scenario, draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return self.forecast_marshall.load(MKeys.beta_scales(scenario=scenario, draw_id=draw_id))

    def save_raw_outputs(self, raw_outputs: pd.DataFrame, scenario: str, draw_id: int, strict: bool):
        self.forecast_marshall.dump(raw_outputs,
                                    key=MKeys.forecast_raw_outputs(scenario=scenario, draw_id=draw_id),
                                    strict=strict)

    def load_raw_outputs(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return self.forecast_marshall.load(key=MKeys.forecast_raw_outputs(scenario=scenario, draw_id=draw_id))

    def save_resampling_map(self, resampling_map):
        with (self.forecast_paths.root_dir / 'resampling_map.yaml').open('w') as map_file:
            yaml.dump(resampling_map, map_file)

    def load_resampling_map(self):
        with (self.forecast_paths.root_dir / 'resampling_map.yaml').open() as map_file:
            resampling_map = yaml.full_load(map_file)
        return resampling_map

    def save_reimposition_dates(self, reimposition_dates: pd.DataFrame, scenario: str, reimposition_number: int):
        self.forecast_marshall.dump(reimposition_dates,
                                    key=MKeys.reimposition_dates(scenario=scenario,
                                                                 reimposition_number=reimposition_number))

    def load_reimposition_dates(self, scenario: str, reimposition_number: int):
        return self.forecast_marshall.load(key=MKeys.reimposition_dates(scenario=scenario,
                                                                        reimposition_number=reimposition_number))

    def save_output_draws(self, output_draws: pd.DataFrame, scenario: str, measure: str) -> None:
        self.postprocessing_marshall.dump(output_draws,
                                          key=MKeys.forecast_output_draws(scenario=scenario, measure=measure))

    def load_output_draws(self, scenario: str, measure: str) -> pd.DataFrame:
        return self.postprocessing_marshall.load(key=MKeys.forecast_output_draws(scenario=scenario, measure=measure))

    def save_output_summaries(self, output_summaries: pd.DataFrame, scenario: str, measure: str) -> None:
        self.postprocessing_marshall.dump(output_summaries,
                                          key=MKeys.forecast_output_summaries(scenario=scenario, measure=measure))

    def load_output_summaries(self, scenario: str, measure: str) -> pd.DataFrame:
        return self.postprocessing_marshall.load(key=MKeys.forecast_output_summaries(scenario=scenario,
                                                                                     measure=measure))

    def save_output_miscellaneous(self, output_miscellaneous: pd.DataFrame, scenario: str, measure: str) -> None:
        self.postprocessing_marshall.dump(output_miscellaneous,
                                          key=MKeys.forecast_output_miscellaneous(scenario=scenario, measure=measure))

    def load_output_miscellaneous(self, scenario: str, measure: str) -> pd.DataFrame:
        return self.postprocessing_marshall.load(key=MKeys.forecast_output_miscellaneous(scenario=scenario,
                                                                                         measure=measure))






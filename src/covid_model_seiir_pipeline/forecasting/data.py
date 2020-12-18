from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline import io
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification, ScenarioSpecification


# TODO: move data interfaces up a package level and fuse with regression data interface.

class ForecastDataInterface:

    def __init__(self,
                 regression_root: io.RegressionRoot,
                 covariate_root: io.CovariateRoot,
                 forecast_root: io.ForecastRoot):
        self.regression_root = regression_root
        self.covariate_root = covariate_root
        self.forecast_root = forecast_root

    @classmethod
    def from_specification(cls, specification: ForecastSpecification) -> 'ForecastDataInterface':
        # TODO: specify input format from config
        regression_root = io.RegressionRoot(specification.data.regression_version)
        covariate_root = io.CovariateRoot(specification.data.covariate_version)
        # TODO: specify output format from config.
        forecast_root = io.ForecastRoot(specification.data.output_root)

        return cls(
            regression_root=regression_root,
            covariate_root=covariate_root,
            forecast_root=forecast_root,
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.forecast_root, **prefix_args)

    def get_n_draws(self) -> int:
        regression_spec = io.load(self.regression_root.specification())
        return regression_spec['parameters']['n_draws']

    def load_location_ids(self) -> List[int]:
        return io.load(self.regression_root.locations())

    def load_thetas(self, theta_specification: Union[str, int]) -> pd.Series:
        location_ids = self.load_location_ids()
        if isinstance(theta_specification, str):
            thetas = pd.read_csv(theta_specification).set_index('location_id')['theta']
            thetas = thetas.reindex(location_ids, fill_value=0)
        else:
            thetas = pd.Series(theta_specification,
                               index=pd.Index(location_ids, name='location_id'),
                               name='theta')
        return thetas

    def check_covariates(self, scenarios: Dict[str, ScenarioSpecification]) -> List[str]:
        regression_spec = io.load(self.regression_root.specification())
        # Bit of a hack.
        forecast_version = str(self.covariate_root._root)
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
                if not io.exists(self.covariate_root[covariate](covariate_scenario=covariate_version)):
                    raise FileNotFoundError(f'No {covariate_version} file found for covariate {covariate}.')
        return list(regression_covariates)

    def get_infectionator_metadata(self):
        regression_spec = io.load(self.regression_root.specification())
        infection_root = io.InfectionRoot(regression_spec['data']['infection_version'])
        return io.load(infection_root.metadata())

    def load_full_data(self):
        metadata = self.get_infectionator_metadata()
        # TODO: metadata abstraction?
        model_inputs_version = metadata['death']['metadata']['model_inputs_metadata']['output_path']
        full_data_path = Path(model_inputs_version) / 'full_data.csv'
        full_data = pd.read_csv(full_data_path)
        full_data['date'] = pd.to_datetime(full_data['Date'])
        full_data = full_data.drop(columns=['Date'])
        return full_data

    def load_population(self):
        metadata = self.get_infectionator_metadata()
        # TODO: metadata abstraction?
        model_inputs_version = metadata['death']['metadata']['model_inputs_metadata']['output_path']
        population_path = Path(model_inputs_version) / 'output_measures' / 'population' / 'all_populations.csv'
        population_data = pd.read_csv(population_path)
        return population_data

    def load_ifr_data(self):
        metadata = self.get_infectionator_metadata()
        # TODO: metadata abstraction?
        ifr_version = metadata['run_arguments']['ifr_custom_path']
        data_path = Path(ifr_version) / 'terminal_ifr.csv'
        data = pd.read_csv(data_path)
        return data.set_index('location_id')

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
        return io.load(self.regression_root.coefficients(draw_id=draw_id))

    # TODO: inverse is RegressionDataInterface.save_date_file
    def load_transition_date(self, draw_id: int) -> pd.Series:
        dates_df = io.load(self.regression_root.dates(draw_id=draw_id))
        dates_df['end_date'] = pd.to_datetime(dates_df['end_date'])
        transition_date = dates_df.set_index('location_id').sort_index()['end_date'].rename('date')
        return transition_date

    # TODO: inverse is RegressionDataInterface.save_regression_betas
    def load_beta_regression(self, draw_id: int) -> pd.DataFrame:
        beta_regression = io.load(self.regression_root.beta(draw_id=draw_id))
        beta_regression['date'] = pd.to_datetime(beta_regression['date'])
        return beta_regression

    # TODO: inverse is RegressionDataInterface.save_location_data
    def load_infection_data(self, draw_id: int) -> pd.DataFrame:
        infection_data = io.load(self.regression_root.data(draw_id=draw_id))
        infection_data['date'] = pd.to_datetime(infection_data['date'])
        return infection_data

    def load_covariate(self, covariate: str, covariate_version: str, location_ids: List[int],
                       with_observed: bool = False) -> pd.DataFrame:
        covariate_df = io.load(self.covariate_root[covariate](covariate_scenario=covariate_version))
        covariate_df = self._format_covariate_data(covariate_df, location_ids, with_observed)
        covariate_df = (covariate_df
                        .rename(columns={f'{covariate}_{covariate_version}': covariate})
                        .filter(columns=[covariate]))
        return covariate_df

    def load_covariates(self, scenario: ScenarioSpecification, location_ids: List[int]) -> pd.DataFrame:
        covariate_data = []
        for covariate, covariate_version in scenario.covariates.items():
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate, covariate_version, location_ids))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data.reset_index()

    def load_mobility_info(self, info_type: str, location_ids: List[int]):
        info_df = io.load(self.covariate_root.mobility_info(info_type=info_type))
        return self._format_covariate_data(info_df, location_ids)

    def load_vaccine_info(self, info_type: str, location_ids: List[int]):
        info_df = io.load(self.covariate_root.vaccine_info(info_type=info_type))
        return self._format_covariate_data(info_df, location_ids)

    def _format_covariate_data(self, dataset: pd.DataFrame, location_ids: List[int], with_observed: bool = False):
        index_columns = ['location_id']
        if with_observed:
            index_columns.append('observed')
        dataset = dataset.loc[dataset['location_id'].isin(location_ids), :]
        if 'date' in dataset.columns:
            dataset['date'] = pd.to_datetime(dataset['date'])
            index_columns.append('date')
        return dataset.set_index(index_columns)

    def load_scenario_specific_data(self,
                                    location_ids: List[int],
                                    scenario_spec: ScenarioSpecification) -> 'ScenarioData':
        if scenario_spec.system == 'vaccine':
            forecast_scenario = scenario_spec.system_params.get('forecast_version', 'reference')
            vaccinations = self.load_vaccine_info(
                f'vaccinations_{forecast_scenario}',
                location_ids,
            )
        else:
            vaccinations = None

        if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
            percent_mandates = self.load_mobility_info('mandate_lift', location_ids)
            mandate_effects = self.load_mobility_info('effect', location_ids)
        else:
            percent_mandates = None
            mandate_effects = None

        scenario_data = ScenarioData(
            vaccinations=vaccinations,
            percent_mandates=percent_mandates,
            mandate_effects=mandate_effects
        )
        return scenario_data

    def save_raw_covariates(self, covariates: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(covariates, self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))

    def load_beta_params(self, draw_id: int) -> Dict[str, float]:
        df = io.load(self.regression_root.parameters(draw_id=draw_id))
        return df.set_index('params')['values'].to_dict()

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(forecasts, self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))

    def save_beta_scales(self, scales: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(scales, self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def save_raw_outputs(self, raw_outputs: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(raw_outputs, self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    def save_resampling_map(self, resampling_map):
        io.dump(resampling_map, self.forecast_root.resampling_map())




@dataclass
class ScenarioData:
    vaccinations: Optional[pd.DataFrame]
    percent_mandates: Optional[pd.DataFrame]
    mandate_effects: Optional[pd.DataFrame]

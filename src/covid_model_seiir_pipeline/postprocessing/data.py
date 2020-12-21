from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline import io, static_vars, utilities
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.postprocessing.specification import (
    PostprocessingSpecification,
    AggregationSpecification,
)


class PostprocessingDataInterface:

    def __init__(self,
                 forecast_root: io.ForecastRoot,
                 postprocessing_root: io.PostprocessingRoot):
        self.forecast_root = forecast_root
        self.postprocessing_root = postprocessing_root

    @classmethod
    def from_specification(cls, specification: PostprocessingSpecification):
        forecast_root = io.ForecastRoot(specification.data.forecast_version)
        postprocessing_root = io.PostprocessingRoot(specification.data.output_root)

        return cls(
            forecast_root=forecast_root,
            postprocessing_root=postprocessing_root,
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.postprocessing_root, **prefix_args)

    ################
    # Data loaders #
    ################

    def get_n_draws(self) -> int:
        return self._get_forecast_data_inteface().get_n_draws()

    def get_covariate_names(self, scenarios: List[str]) -> List[str]:
        forecast_spec = ForecastSpecification.from_dict(io.load(self.forecast_root.specification()))
        forecast_di = ForecastDataInterface.from_specification(forecast_spec)
        scenarios = {scenario: spec for scenario, spec in forecast_spec.scenarios.items() if scenario in scenarios}
        return forecast_di.check_covariates(scenarios)

    def get_covariate_version(self, covariate_name: str, scenario: str) -> str:
        forecast_spec = ForecastSpecification.from_dict(io.load(self.forecast_root.specification()))
        return forecast_spec.scenarios[scenario].covariates[covariate_name]

    def load_location_ids(self):
        return self._get_forecast_data_inteface().load_location_ids()

    def load_regression_coefficients(self, draw_id: int) -> pd.Series:
        coefficients = self._get_forecast_data_inteface().load_regression_coefficients(draw_id)
        coefficients = coefficients.set_index('location_id').stack().reset_index()
        coefficients.columns = ['location_id', 'covariate', draw_id]
        coefficients = coefficients.set_index(['location_id', 'covariate'])[draw_id]
        return coefficients

    def load_scaling_parameters(self, draw_id: int, scenario: str) -> pd.Series:
        scaling_parameters = self._get_forecast_data_inteface().load_beta_scales(scenario, draw_id)
        scaling_parameters = scaling_parameters.set_index('location_id').stack().reset_index()
        scaling_parameters.columns = ['location_id', 'scaling_parameter', draw_id]
        scaling_parameters = scaling_parameters.set_index(['location_id', 'scaling_parameter'])[draw_id]
        return scaling_parameters

    def load_covariate(self, draw_id: int, covariate: str, time_varying: bool,
                       scenario: str, with_observed: bool = False) -> pd.Series:
        covariates = io.load(self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))
        covariates['date'] = pd.to_datetime(covariates['date'])
        covariates = covariates.set_index(['location_id', 'date']).sort_index()
        if time_varying:
            covariate = covariates[covariate].rename(draw_id)
        else:
            covariate = covariates.groupby(level='location_id')[covariate].max().rename(draw_id)
        return covariate

    def load_input_covariate(self, covariate: str, covariate_version: str, location_ids: List[int]):
        return self._get_forecast_data_inteface().load_covariate(covariate, covariate_version,
                                                                 location_ids, with_observed=True)

    def load_betas(self, draw_id: int, scenario: str) -> pd.Series:
        ode_params = io.load(self.forecast_root.ode_params(scenario=scenario, draw_id=draw_id))
        ode_params['date'] = pd.to_datetime(ode_params['date'])
        betas = (ode_params
                 .set_index(['location_id', 'date'])['beta']
                 .sort_index()
                 .rename(draw_id))
        return betas

    def load_beta_residuals(self, draw_id: int) -> pd.Series:
        beta_regression = self._get_forecast_data_inteface().load_beta_regression(draw_id)
        beta_regression = (beta_regression
                           .set_index(['location_id', 'date'])
                           .sort_index()[['beta', 'beta_pred']])
        beta_residual = np.log(beta_regression['beta'] / beta_regression['beta_pred']).rename(draw_id)
        return beta_residual

    def load_raw_outputs(self, draw_id: int, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        draw_df = io.load(self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))
        draw_df = draw_df.set_index(['location_id', 'date']).sort_index()
        deaths = draw_df.reset_index().set_index(['location_id', 'date', 'observed'])['deaths'].rename(draw_id)
        infections = draw_df['infections'].rename(draw_id)
        r_effective = draw_df['r_effective'].rename(draw_id)
        return deaths, infections, r_effective

    def load_elastispliner_inputs(self) -> pd.DataFrame:
        metadata = self._get_forecast_data_inteface().get_infectionator_metadata()
        deaths_version = Path(metadata['death']['metadata']['output_path'])
        es_inputs = pd.read_csv(deaths_version / 'model_data.csv')
        es_inputs['date'] = pd.to_datetime(es_inputs['date'])
        es_inputs = es_inputs.set_index(['location_id', 'date'])
        cumulative_cases = (es_inputs['Confirmed case rate'] * es_inputs['population']).rename('cumulative_cases')
        cumulative_deaths = (es_inputs['Death rate'] * es_inputs['population']).rename('cumulative_deaths')
        cumulative_hospitalizations = (es_inputs['Hospitalization rate'] * es_inputs['population'])
        cumulative_hospitalizations = cumulative_hospitalizations.rename('cumulative_hospitalizations')

        death_max_date = cumulative_deaths.dropna().reset_index().groupby('location_id')['date'].max()
        case_max_date = cumulative_cases.dropna().reset_index().groupby('location_id')['date'].max()
        hosp_max_date = cumulative_hospitalizations.dropna().reset_index().groupby('location_id')['date'].max()

        case_shift = (death_max_date - case_max_date).dt.days.max()  # Constant across locations
        hosp_shift = (death_max_date - hosp_max_date).dt.days.max()  # Constant across locations

        cumulative_cases = cumulative_cases.groupby('location_id').shift(case_shift)
        cumulative_hospitalizations = cumulative_hospitalizations.groupby('location_id').shift(hosp_shift)
        es_inputs = pd.concat([cumulative_cases, cumulative_deaths, cumulative_hospitalizations], axis=1)
        es_inputs = es_inputs.loc[es_inputs.notnull().any(axis=1)]
        return es_inputs

    def load_elastispliner_outputs(self, noisy: bool) -> pd.DataFrame:
        metadata = self._get_forecast_data_inteface().get_infectionator_metadata()
        deaths_version = Path(metadata['death']['metadata']['output_path'])
        es_file = 'model_results.csv' if noisy else 'model_results_refit.csv'
        es_outputs = pd.read_csv(deaths_version / es_file)
        es_outputs = es_outputs.set_index(['location_id', 'date', 'observed'])
        n_draws = self.get_n_draws()
        es_outputs = es_outputs.rename(columns={f'draw_{i}': i for i in range(n_draws)})
        es_outputs = es_outputs.groupby(level='location_id').apply(lambda x: x - x.shift(fill_value=0))
        return es_outputs

    def load_full_data(self) -> pd.DataFrame:
        full_data = self._get_forecast_data_inteface().load_full_data()
        full_data = full_data.set_index(['location_id', 'date'])
        full_data = full_data.rename(columns={
            'Deaths': 'cumulative_deaths',
            'Confirmed': 'cumulative_cases',
            'Hospitalizations': 'cumulative_hospitalizations',
        })
        full_data = full_data[['cumulative_cases', 'cumulative_deaths', 'cumulative_hospitalizations']]
        return full_data

    def build_version_map(self) -> pd.Series:
        forecast_di = self._get_forecast_data_inteface()
        version_map = {
            'postprocessing_version': Path(self.postprocessing_root._root).name,
            'forecast_version': Path(forecast_di.forecast_root._root).name,
            'regression_version': Path(forecast_di.regression_root._root).name,
            'covariate_version': Path(forecast_di.covariate_root._root).name
        }

        # FIXME: infectionator doesn't do metadata the right way.
        inf_metadata = forecast_di.get_infectionator_metadata()
        inf_output_dir = inf_metadata['wrapped_R_call'][-1].split()[1].strip("'")
        version_map['infectionator_version'] = Path(inf_output_dir).name

        death_metadata = inf_metadata['death']['metadata']
        version_map['elastispliner_version'] = Path(death_metadata['output_path']).name

        model_inputs_metadata = death_metadata['model_inputs_metadata']
        version_map['model_inputs_version'] = Path(model_inputs_metadata['output_path']).name

        snapshot_metadata = model_inputs_metadata['snapshot_metadata']
        version_map['snapshot_version'] = Path(snapshot_metadata['output_path']).name
        jhu_snapshot_metadata = model_inputs_metadata['jhu_snapshot_metadata']
        version_map['jhu_snapshot_version'] = Path(jhu_snapshot_metadata['output_path']).name
        try:
            # There is a typo in the process that generates this key.
            # Protect ourselves in case they fix it without warning.
            webscrape_metadata = model_inputs_metadata['webcrape_metadata']
        except KeyError:
            webscrape_metadata = model_inputs_metadata['webscrape_metadata']
        version_map['webscrape_version'] = Path(webscrape_metadata['output_path']).name

        version_map['location_set_version_id'] = model_inputs_metadata['run_arguments']['lsvid']
        try:
            version_map['location_set_version_id'] = int(version_map['location_set_version_id'])
        except:
            pass
        version_map['data_date'] = Path(snapshot_metadata['output_path']).name.split('.')[0].replace('_', '-')

        version_map = pd.Series(version_map)
        version_map = version_map.reset_index()
        version_map.columns = ['name', 'version']
        return version_map

    def load_populations(self) -> pd.DataFrame:
        metadata = self._get_forecast_data_inteface().get_infectionator_metadata()
        model_inputs_path = Path(
            metadata['death']['metadata']['model_inputs_metadata']['output_path']
        )
        population_path = model_inputs_path / 'output_measures' / 'population' / 'all_populations.csv'
        populations = pd.read_csv(population_path)
        return populations

    def load_hierarchy(self) -> pd.DataFrame:
        metadata = self._get_forecast_data_inteface().get_infectionator_metadata()
        model_inputs_path = Path(
            metadata['death']['metadata']['model_inputs_metadata']['output_path']
        )
        hierarchy_path = model_inputs_path / 'locations' / 'modeling_hierarchy.csv'
        hierarchy = pd.read_csv(hierarchy_path)
        return hierarchy

    def load_aggregation_heirarchy(self, aggregation_spec: AggregationSpecification):
        if any(aggregation_spec.to_dict().values()):
            return utilities.load_location_hierarchy(**aggregation_spec.to_dict())
        else:
            return self.load_hierarchy()

    def get_locations_modeled_and_missing(self):
        hierarchy = self.load_hierarchy()
        modeled_locations = self._get_forecast_data_inteface().load_location_ids()
        most_detailed_locs = hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].unique().tolist()
        missing_locations = list(set(most_detailed_locs).difference(modeled_locations))
        locations_modeled_and_missing = {'modeled': modeled_locations, 'missing': missing_locations}
        return locations_modeled_and_missing

    def load_specification(self) -> 'PostprocessingSpecification':
        spec_dict = io.load(self.postprocessing_root.specification())
        return PostprocessingSpecification.from_dict(spec_dict)

    def load_resampling_map(self) -> Dict[int, Dict[str, List[int]]]:
        return io.load(self.postprocessing_root.resampling_map())

    def load_output_draws(self, scenario: str, measure: str) -> pd.DataFrame:
        return io.load(self.postprocessing_root.output_draws(scenario=scenario, measure=measure))

    def load_output_summaries(self, scenario: str, measure: str) -> pd.DataFrame:
        return io.load(self.postprocessing_root.output_summaries(scenario=scenario, measure=measure))

    def load_output_miscellaneous(self, scenario: str, measure: str, is_table: bool) -> pd.DataFrame:
        key = self._get_output_miscellaneous_key(scenario, measure, is_table)
        return io.load(key)

    def load_previous_version_output_draws(self, version: str, scenario: str, measure: str):
        previous_di = self._get_previous_version_data_interface(version)
        return previous_di.load_output_draws(scenario, measure)

    ################
    # Data writers #
    ################

    def save_specification(self, specification: PostprocessingSpecification):
        io.dump(specification.to_dict(), self.postprocessing_root.specification())

    def save_resampling_map(self, resampling_map: Dict[int, Dict[str, List[int]]]):
        io.dump(resampling_map, self.postprocessing_root.resampling_map())

    def save_output_draws(self, output_draws: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_draws, self.postprocessing_root.output_draws(scenario=scenario, measure=measure))

    def save_output_summaries(self, output_summaries: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_summaries, self.postprocessing_root.output_summaries(scenario=scenario, measure=measure))

    def save_output_miscellaneous(self, output_miscellaneous,
                                  scenario: str, measure: str, is_table: bool):
        key = self._get_output_miscellaneous_key(scenario, measure, is_table)
        io.dump(output_miscellaneous, key)

    #########################
    # Non-interface methods #
    #########################

    def _get_forecast_data_inteface(self):
        forecast_spec = ForecastSpecification.from_dict(io.load(self.forecast_root.specification()))
        forecast_di = ForecastDataInterface.from_specification(forecast_spec)
        return forecast_di

    def _get_previous_version_data_interface(self, version: str) -> 'PostprocessingDataInterface':
        previous_spec_path = Path(version) / static_vars.POSTPROCESSING_SPECIFICATION_FILE
        previous_spec = PostprocessingSpecification.from_path(previous_spec_path)
        previous_di = PostprocessingDataInterface.from_specification(previous_spec)
        return previous_di

    def _get_output_miscellaneous_key(self, scenario: str, measure: str, is_table: bool):
        if is_table:
            key = self.postprocessing_root.output_miscellaneous(scenario=scenario, measure=measure)
        else:
            key = io.MetadataKey(root=self.postprocessing_root._root / scenario / 'output_miscellaneous',
                                 disk_format=self.postprocessing_root._metadata_format,
                                 data_type=measure)
        return key

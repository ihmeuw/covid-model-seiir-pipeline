from pathlib import Path
from typing import Dict, List, Iterable, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.forecasting import (
    ForecastSpecification,
    ForecastDataInterface,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.specification import (
    PostprocessingSpecification,
    AggregationSpecification,
)


class PostprocessingDataInterface:

    def __init__(self,
                 forecast_data_interface: ForecastDataInterface,
                 postprocessing_root: io.PostprocessingRoot):
        self.forecast_data_interface = forecast_data_interface
        self.postprocessing_root = postprocessing_root

    @classmethod
    def from_specification(cls, specification: PostprocessingSpecification):
        forecast_spec = ForecastSpecification.from_version_root(specification.data.seir_forecast_version)
        forecast_data_interface = ForecastDataInterface.from_specification(forecast_spec)
        postprocessing_root = io.PostprocessingRoot(specification.data.output_root)

        return cls(
            forecast_data_interface=forecast_data_interface,
            postprocessing_root=postprocessing_root,
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.postprocessing_root, **prefix_args)

    def get_n_draws(self) -> int:
        return self.forecast_data_interface.get_n_draws()

    ####################
    # Prior Stage Data #
    ####################

    def load_hierarchy(self, name: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_hierarchy(name=name)

    def load_population(self, measure: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_population(measure=measure)

    def load_reported_epi_data(self) -> pd.DataFrame:
        return self.forecast_data_interface.load_reported_epi_data()

    def load_hospital_census_data(self) -> pd.DataFrame:
        return self.forecast_data_interface.load_hospital_census_data()

    def load_hospital_bed_capacity(self) -> pd.DataFrame:
        return self.forecast_data_interface.load_hospital_bed_capacity()

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_total_covid_scalars(draw_id=draw_id)

    def load_seroprevalence(self, draw_id: int = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_seroprevalence(draw_id=draw_id)

    def load_sensitivity(self, draw_id: int = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_sensitivity(draw_id)

    def load_testing_data(self) -> pd.DataFrame:
        return self.forecast_data_interface.load_testing_data()


    def get_covariate_version(self, covariate_name: str, scenario: str) -> str:
        forecast_spec = self.forecast_data_interface.load_specification()
        return forecast_spec.scenarios[scenario].covariates[covariate_name]

    def load_covariate(self, draw_id: int, covariate: str, time_varying: bool,
                       scenario: str, with_observed: bool = False) -> pd.Series:
        covariates = self.forecast_data_interface.load_raw_covariates(scenario, draw_id)
        if time_varying:
            covariate = covariates[covariate].rename(draw_id)
        else:
            covariate = covariates.groupby(level='location_id')[covariate].max().rename(draw_id)
        return covariate

    def load_input_covariate(self, covariate: str, covariate_version: str):
        return self.forecast_data_interface.load_covariate(covariate, covariate_version, with_observed=True)

    def load_covariates(self, covariates: Iterable[str]) -> pd.DataFrame:
        return self.forecast_data_interface.load_covariates(covariates)

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_covariate_info(covariate, info_type)

    def load_mandate_data(self, mobility_scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.forecast_data_interface.load_mandate_data(mobility_scenario)

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_variant_prevalence(scenario)

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_waning_parameters(measure)

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_vaccine_uptake(scenario)

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_vaccine_risk_reduction(scenario)

    def load_covariate_options(self, draw_id: int = None) -> Dict:
        return self.forecast_data_interface.load_covariate_options(draw_id)

    def load_regression_ode_params(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_regression_ode_params(draw_id)

    def load_phis(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_phis(draw_id)

    def load_input_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_input_epi_measures(draw_id, columns)

    def load_rates_data(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_rates_data(draw_id, columns)

    def load_rates(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_rates(draw_id, columns)

    def load_posterior_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_posterior_epi_measures(draw_id, columns)

    def load_past_compartments(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_past_compartments(draw_id, columns)

    def load_fit_beta(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_fit_beta(draw_id, columns)

    def load_final_seroprevalence(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_final_seroprevalence(draw_id, columns)

    def load_summary(self, measure: str) -> pd.DataFrame:
        return self.forecast_data_interface.load_summary(measure)

    def load_location_ids(self) -> List[int]:
        return self.forecast_data_interface.load_location_ids()

    def load_regression_beta(self, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_regression_beta(draw_id=draw_id)

    def load_coefficients(self, draw_id: int) -> pd.DataFrame:
        coefficients = self.forecast_data_interface.load_coefficients(draw_id)
        coefficients = coefficients.stack().reset_index()
        coefficients.columns = ['location_id', 'covariate', draw_id]
        coefficients = coefficients.set_index(['location_id', 'covariate'])[draw_id]
        return coefficients

    def load_raw_covariates(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return self.forecast_data_interface.load_raw_covariates(scenario, draw_id)

    def load_ode_params(self, scenario: str, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.forecast_data_interface.load_ode_params(scenario, draw_id, columns=columns)

    def load_single_ode_param(self, draw_id: int, scenario: str, measure: str) -> pd.Series:
        draw_df = self.load_ode_params(draw_id=draw_id, scenario=scenario, columns=[measure])
        return draw_df[measure].rename(draw_id)

    def load_components(self, scenario: str, draw_id: int):
        return self.forecast_data_interface.load_components(scenario, draw_id)

    def load_beta_scales(self, scenario: str, draw_id: int):
        scaling_parameters = self.forecast_data_interface.load_beta_scales(scenario, draw_id)
        scaling_parameters = scaling_parameters.stack().reset_index()
        scaling_parameters.columns = ['location_id', 'scaling_parameter', draw_id]
        scaling_parameters = scaling_parameters.set_index(['location_id', 'scaling_parameter'])[draw_id]
        return scaling_parameters

    def load_beta_residual(self, scenario: str, draw_id: int):
        return self.forecast_data_interface.load_beta_residual(scenario, draw_id)

    def load_raw_outputs(self, scenario: str, draw_id: int, columns: List[str] = None):
        return self.forecast_data_interface.load_raw_outputs(scenario, draw_id, columns=columns)

    def load_single_raw_output(self, draw_id: int, scenario: str, measure: str) -> pd.Series:
        # FIXME: HACK HACK HACK.  Data alignment problems I can't chase right now.
        idx = self.load_raw_outputs(scenario=scenario, draw_id=0, columns=[measure]).index
        draw_df = self.load_raw_outputs(scenario=scenario, draw_id=draw_id, columns=[measure])
        return draw_df[measure].reindex(idx).rename(draw_id)

    def load_raw_output_deaths(self, draw_id: int, scenario: str) -> pd.Series:
        draw_df = self.load_raw_outputs(scenario=scenario, draw_id=draw_id, columns=['modeled_deaths_total'])
        draw_df = draw_df.groupby('location_id').bfill().groupby('location_id').ffill()
        draw_df = draw_df.modeled_deaths_total.rename(draw_id)
        return draw_df

    def load_beta_residuals(self, draw_id: int, scenario: str) -> pd.Series:
        beta_residual = self.forecast_data_interface.load_beta_residual(scenario=scenario, draw_id=draw_id)
        beta_residual = beta_residual.set_index(['location_id', 'date'])['log_beta_residual'].rename(draw_id)
        return beta_residual

    def load_scaled_beta_residuals(self, draw_id: int, scenario: str) -> pd.Series:
        beta_residual = self.forecast_data_interface.load_beta_residual(scenario=scenario, draw_id=draw_id)
        beta_residual = beta_residual.set_index(['location_id', 'date'])['scaled_log_beta_residual'].rename(draw_id)
        return beta_residual

    #########################
    # Forecast data loaders #
    #########################
    #
    # def load_full_data_unscaled(self) -> pd.DataFrame:
    #     return self._get_forecast_data_inteface().load_full_data_unscaled()

    # def get_covariate_names(self, scenarios: List[str]) -> List[str]:
    #     forecast_spec = ForecastSpecification.from_dict(io.load(self.forecast_root.specification()))
    #     forecast_di = ForecastDataInterface.from_specification(forecast_spec)
    #     scenarios = {scenario: spec for scenario, spec in forecast_spec.scenarios.items() if scenario in scenarios}
    #     return forecast_di.check_covariates(scenarios)


    # def load_vaccination_summaries(self, measure: str):
    #     return self.forecast_data_interface.load_vaccination_summaries(
    #         measure,
    #     )

    # def load_vaccine_efficacy(self):
    #     return self._get_forecast_data_inteface().load_vaccine_efficacy()
    #
    # def load_raw_variant_prevalence(self) -> pd.DataFrame:
    #     return self._get_forecast_data_inteface().load_variant_prevalence('reference')


    ##############################
    # Miscellaneous data loaders #
    ##############################

    # def build_version_map(self) -> pd.Series:
    #     forecast_di = self.forecast_data_interface
    #     version_map = {
    #         'postprocessing_version': Path(self.postprocessing_root._root).name,
    #         'forecast_version': Path(forecast_di.forecast_root._root).name,
    #         'regression_version': Path(forecast_di.regression_root._root).name,
    #         'covariate_version': Path(forecast_di.covariate_root._root).name
    #     }
    #
    #     inf_metadata = forecast_di.get_infections_metadata()
    #     version_map['infections_version'] = Path(inf_metadata['output_path']).name
    #
    #     model_inputs_metadata = inf_metadata['model_inputs_metadata']
    #     version_map['model_inputs_version'] = Path(model_inputs_metadata['output_path']).name
    #
    #     snapshot_metadata = model_inputs_metadata['snapshot_metadata']
    #     version_map['snapshot_version'] = Path(snapshot_metadata['output_path']).name
    #     jhu_snapshot_metadata = model_inputs_metadata['jhu_snapshot_metadata']
    #     version_map['jhu_snapshot_version'] = Path(jhu_snapshot_metadata['output_path']).name
    #     try:
    #         # There is a typo in the process that generates this key.
    #         # Protect ourselves in case they fix it without warning.
    #         webscrape_metadata = model_inputs_metadata['webcrape_metadata']
    #     except KeyError:
    #         webscrape_metadata = model_inputs_metadata['webscrape_metadata']
    #     version_map['webscrape_version'] = Path(webscrape_metadata['output_path']).name
    #
    #     version_map['location_set_version_id'] = model_inputs_metadata['run_arguments']['lsvid']
    #     try:
    #         version_map['location_set_version_id'] = int(version_map['location_set_version_id'])
    #     except:
    #         pass
    #     version_map['data_date'] = Path(snapshot_metadata['output_path']).name.split('.')[0].replace('_', '-')
    #
    #     version_map = pd.Series(version_map)
    #     version_map = version_map.reset_index()
    #     version_map.columns = ['name', 'version']
    #     return version_map

    def load_aggregation_hierarchy(self, aggregation_spec: AggregationSpecification):
        if any(aggregation_spec.to_dict().values()):
            return utilities.load_location_hierarchy(**aggregation_spec.to_dict())
        else:
            return self.load_hierarchy('pred')

    def get_locations_modeled_and_missing(self):
        hierarchy = self.load_hierarchy('pred')
        modeled_locations = set(self.forecast_data_interface.load_location_ids())
        spec = self.load_specification()
        spliced_locations = set([location for splicing_spec in spec.splicing for location in splicing_spec.locations])
        included_locations = list(modeled_locations | spliced_locations)

        most_detailed_locs = hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].unique().tolist()
        missing_locations = list(set(most_detailed_locs).difference(included_locations))
        locations_modeled_and_missing = {'modeled': included_locations, 'missing': missing_locations}
        return locations_modeled_and_missing
    #
    # def load_hospital_correction_factors(self):
    #     return self._get_forecast_data_inteface().load_hospital_correction_factors().to_df()

    ###########################
    # Postprocessing data I/O #
    ###########################

    def save_specification(self, specification: PostprocessingSpecification):
        io.dump(specification.to_dict(), self.postprocessing_root.specification())

    def load_specification(self) -> 'PostprocessingSpecification':
        spec_dict = io.load(self.postprocessing_root.specification())
        return PostprocessingSpecification.from_dict(spec_dict)

    def save_resampling_map(self, resampling_map: Dict[int, Dict[str, List[int]]]):
        io.dump(resampling_map, self.postprocessing_root.resampling_map())

    def load_resampling_map(self) -> Dict[int, Dict[str, List[int]]]:
        return io.load(self.postprocessing_root.resampling_map())

    def save_output_draws(self, output_draws: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_draws, self.postprocessing_root.output_draws(scenario=scenario, measure=measure))

    def load_output_draws(self, scenario: str, measure: str) -> pd.DataFrame:
        return io.load(self.postprocessing_root.output_draws(scenario=scenario, measure=measure))

    def load_previous_version_output_draws(self, version: str, scenario: str, measure: str):
        previous_di = self._get_previous_version_data_interface(version)
        return previous_di.load_output_draws(scenario, measure)

    def save_output_summaries(self, output_summaries: pd.DataFrame, scenario: str, measure: str):
        io.dump(output_summaries, self.postprocessing_root.output_summaries(scenario=scenario, measure=measure))

    def load_output_summaries(self, scenario: str, measure: str) -> pd.DataFrame:
        return io.load(self.postprocessing_root.output_summaries(scenario=scenario, measure=measure))

    def save_output_miscellaneous(self, output_miscellaneous,
                                  scenario: str, measure: str, is_table: bool):
        key = self._get_output_miscellaneous_key(scenario, measure, is_table)
        io.dump(output_miscellaneous, key)

    def load_output_miscellaneous(self, scenario: str, measure: str, is_table: bool) -> pd.DataFrame:
        key = self._get_output_miscellaneous_key(scenario, measure, is_table)
        return io.load(key)

    #########################
    # Non-interface methods #
    #########################

    def _get_previous_version_data_interface(self, version: str) -> 'PostprocessingDataInterface':
        previous_spec = PostprocessingSpecification.from_version_root(version)
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

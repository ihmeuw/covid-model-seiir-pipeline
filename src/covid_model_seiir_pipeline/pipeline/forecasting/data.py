from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple, Union

from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.regression import (
    RegressionDataInterface,
    RegressionSpecification,
    HospitalParameters,
)
from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
    ForecastSpecification,
    ScenarioSpecification,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model import (
    RatioData,
    HospitalMetrics,
    HospitalCorrectionFactors,
    HospitalCensusData,
)


class ForecastDataInterface:

    def __init__(self,
                 regression_root: io.RegressionRoot,
                 covariate_root: io.CovariateRoot,
                 forecast_root: io.ForecastRoot,
                 fh_subnationals: bool):
        self.regression_root = regression_root
        self.covariate_root = covariate_root
        self.forecast_root = forecast_root
        self.fh_subnationals = fh_subnationals

    @classmethod
    def from_specification(cls, specification: ForecastSpecification) -> 'ForecastDataInterface':
        regression_spec_path = Path(specification.data.regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
        regression_spec = RegressionSpecification.from_path(regression_spec_path)
        regression_root = io.RegressionRoot(specification.data.regression_version,
                                            data_format=regression_spec.data.output_format)
        covariate_root = io.CovariateRoot(specification.data.covariate_version)
        # TODO: specify output format from config.
        forecast_root = io.ForecastRoot(specification.data.output_root,
                                        data_format=specification.data.output_format)

        return cls(
            regression_root=regression_root,
            covariate_root=covariate_root,
            forecast_root=forecast_root,
            fh_subnationals=specification.data.fh_subnationals
        )

    def make_dirs(self, **prefix_args):
        io.touch(self.forecast_root, **prefix_args)

    ############################
    # Regression paths loaders #
    ############################

    def get_n_draws(self) -> int:
        return self._get_regression_data_interface().get_n_draws()

    def load_location_ids(self) -> List[int]:
        return self._get_regression_data_interface().load_location_ids()

    def load_betas(self, draw_id: int):
        return self._get_regression_data_interface().load_betas(draw_id=draw_id)

    def load_coefficients(self, draw_id: int) -> pd.DataFrame:
        return self._get_regression_data_interface().load_coefficients(draw_id=draw_id)

    def load_compartments(self, draw_id: int) -> pd.DataFrame:
        return self._get_regression_data_interface().load_compartments(draw_id=draw_id)

    def load_ode_parameters(self, draw_id: int) -> pd.DataFrame:
        return self._get_regression_data_interface().load_ode_parameters(draw_id=draw_id)

    def load_past_infections(self, draw_id: int) -> pd.Series:
        return self._get_regression_data_interface().load_infections(draw_id=draw_id)

    def load_em_scalars(self) -> pd.Series:
        return self._get_regression_data_interface().load_em_scalars()

    def load_past_deaths(self, draw_id: int) -> pd.Series:
        return self._get_regression_data_interface().load_deaths(draw_id=draw_id)

    def get_hospital_parameters(self) -> HospitalParameters:
        return self._get_regression_data_interface().load_specification().hospital_parameters

    def load_hospital_usage(self) -> HospitalMetrics:
        df = self._get_regression_data_interface().load_hospitalizations(measure='usage')
        return HospitalMetrics(**{metric: df[metric] for metric in df.columns})

    def load_hospital_correction_factors(self) -> HospitalCorrectionFactors:
        df = self._get_regression_data_interface().load_hospitalizations(measure='correction_factors')
        return HospitalCorrectionFactors(**{metric: df[metric] for metric in df.columns})

    def load_hospital_census_data(self) -> HospitalCensusData:
        return self._get_regression_data_interface().load_hospital_census_data()

    def load_ifr(self, draw_id: int) -> pd.DataFrame:
        return self._get_regression_data_interface().load_ifr(draw_id=draw_id)

    def load_ihr(self, draw_id: int) -> pd.DataFrame:
        return self._get_regression_data_interface().load_ihr(draw_id=draw_id)

    def load_idr(self, draw_id: int) -> pd.DataFrame:
        return self._get_regression_data_interface().load_idr(draw_id=draw_id)

    def load_ratio_data(self, draw_id: int) -> RatioData:
        return self._get_regression_data_interface().load_ratio_data(draw_id=draw_id)

    ##########################
    # Covariate data loaders #
    ##########################

    def check_covariates(self, scenarios: Dict[str, ScenarioSpecification]) -> List[str]:
        regression_spec = self._get_regression_data_interface().load_specification().to_dict()
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

    def load_covariate(self, covariate: str, covariate_version: str, with_observed: bool = False) -> pd.DataFrame:
        location_ids = self.load_location_ids()
        covariate_df = io.load(self.covariate_root[covariate](covariate_scenario=covariate_version))
        covariate_df = self._format_covariate_data(covariate_df, location_ids, with_observed)
        covariate_df = (covariate_df
                        .rename(columns={f'{covariate}_{covariate_version}': covariate})
                        .loc[:, [covariate]])
        return covariate_df

    def load_covariates(self, scenario: ScenarioSpecification) -> pd.DataFrame:
        covariate_data = []
        for covariate, covariate_version in scenario.covariates.items():
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate, covariate_version))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data

    def load_mobility_info(self, info_type: str):
        location_ids = self.load_location_ids()
        info_df = io.load(self.covariate_root.mobility_info(info_type=info_type))
        return self._format_covariate_data(info_df, location_ids)

    def load_vaccinations(self, vaccine_scenario: str):
        location_ids = self.load_location_ids()
        if vaccine_scenario == 'none':
            # Grab the reference so we get the right index/schema.
            info_df = io.load(self.covariate_root.vaccine_info(info_type='vaccinations_reference'))
            info_df.loc[:, :] = 0.0
        else:
            info_df = io.load(self.covariate_root.vaccine_info(info_type=f'vaccinations_{vaccine_scenario}'))
        return self._format_covariate_data(info_df, location_ids)

    def load_variant_prevalence(self, variant_scenario: str):
        b117_ramp = self.load_covariate('variant_prevalence_non_escape', variant_scenario).variant_prevalence_non_escape
        b117 = self.load_covariate('variant_prevalence_B117', variant_scenario).variant_prevalence_B117
        b1351 = self.load_covariate('variant_prevalence_B1351', variant_scenario).variant_prevalence_B1351
        p1 = self.load_covariate('variant_prevalence_P1', variant_scenario).variant_prevalence_P1
        b1617 = self.load_covariate('variant_prevalence_B1617', variant_scenario).variant_prevalence_B1617.rename('rho_b1617')
        rho_variant = (b1351 + p1 + b1617).rename('rho_variant')
        rho_total = (b117 + rho_variant).rename('rho_total')
        rho = b117_ramp.rename('rho')
        return pd.concat([rho, rho_variant, b1617, rho_total], axis=1)

    #########################
    # Scenario data loaders #
    #########################

    def load_mandate_data(self, mobility_scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        percent_mandates = self.load_mobility_info(f'{mobility_scenario}_mandate_lift')
        mandate_effects = self.load_mobility_info(f'effect')
        return percent_mandates, mandate_effects

    ##############################
    # Miscellaneous data loaders #
    ##############################

    def load_thetas(self, beta_scales: pd.DataFrame) -> pd.Series:
        location_ids = self.load_location_ids()
        thetas = pd.Series(0.0,
                           index=pd.Index(location_ids, name='location_id'),
                           name='theta')
        import pdb; pdb.set_trace()
        return thetas

    def get_infections_metadata(self):
        return self._get_regression_data_interface().get_infections_metadata()

    def get_model_inputs_metadata(self):
        infection_metadata = self.get_infections_metadata()
        return infection_metadata['model_inputs_metadata']

    def load_full_data(self) -> pd.DataFrame:
        metadata = self.get_model_inputs_metadata()
        model_inputs_version = metadata['output_path']
        if self.fh_subnationals:
            full_data_path = Path(model_inputs_version) / 'full_data_fh_subnationals.csv'
        else:
            full_data_path = Path(model_inputs_version) / 'full_data.csv'
        full_data = pd.read_csv(full_data_path)
        full_data['date'] = pd.to_datetime(full_data['Date'])
        full_data = full_data.drop(columns=['Date'])
        full_data['location_id'] = full_data['location_id'].astype(int)
        return full_data

    def load_population(self) -> pd.DataFrame:
        return self._get_regression_data_interface().load_population()

    def load_five_year_population(self) -> pd.DataFrame:
        return self._get_regression_data_interface().load_five_year_population()

    def load_total_deaths(self):
        """Load cumulative deaths by location."""
        location_ids = self.load_location_ids()
        full_data = self.load_full_data()
        total_deaths = full_data.groupby('location_id')['Deaths'].max().rename('deaths')
        return total_deaths.loc[location_ids]

    #####################
    # Forecast data I/O #
    #####################

    def save_specification(self, specification: ForecastSpecification) -> None:
        io.dump(specification.to_dict(), self.forecast_root.specification())

    def load_specification(self) -> ForecastSpecification:
        spec_dict = io.load(self.forecast_root.specification())
        return ForecastSpecification.from_dict(spec_dict)

    def save_raw_covariates(self, covariates: pd.DataFrame, scenario: str, draw_id: int) -> None:
        io.dump(covariates, self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))

    def load_raw_covariates(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.forecast_root.raw_covariates(scenario=scenario, draw_id=draw_id))

    def save_ode_params(self, ode_params: pd.DataFrame, scenario: str, draw_id: int) -> None:
        io.dump(ode_params, self.forecast_root.ode_params(scenario=scenario, draw_id=draw_id))

    def load_ode_params(self, scenario: str, draw_id: int) -> pd.DataFrame:
        return io.load(self.forecast_root.ode_params(scenario=scenario, draw_id=draw_id))

    def save_components(self, forecasts: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(forecasts, self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))

    def load_components(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.component_draws(scenario=scenario, draw_id=draw_id))

    def save_beta_scales(self, scales: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(scales, self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def load_beta_scales(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_scaling(scenario=scenario, draw_id=draw_id))

    def save_beta_residual(self, residual: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(residual, self.forecast_root.beta_residual(scenario=scenario, draw_id=draw_id))

    def load_beta_residual(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.beta_residual(scenario=scenario, draw_id=draw_id))

    def save_raw_outputs(self, raw_outputs: pd.DataFrame, scenario: str, draw_id: int):
        io.dump(raw_outputs, self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    def load_raw_outputs(self, scenario: str, draw_id: int):
        return io.load(self.forecast_root.raw_outputs(scenario=scenario, draw_id=draw_id))

    #########################
    # Non-interface helpers #
    #########################

    @staticmethod
    def _format_covariate_data(dataset: pd.DataFrame, location_ids: List[int], with_observed: bool = False):
        shared_locs = list(set(dataset.index.get_level_values('location_id')).intersection(location_ids))
        dataset = dataset.loc[shared_locs]
        if with_observed:
            dataset = dataset.set_index('observed', append=True)
        return dataset

    def _get_regression_data_interface(self) -> RegressionDataInterface:
        regression_spec = RegressionSpecification.from_dict(io.load(self.regression_root.specification()))
        regression_di = RegressionDataInterface.from_specification(regression_spec)
        return regression_di

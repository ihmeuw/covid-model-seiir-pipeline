from functools import reduce
from typing import Dict, List, Iterable, Optional, Union

from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
)
from covid_model_seiir_pipeline.pipeline.fit import (
    FitSpecification,
    FitDataInterface,
)
from covid_model_seiir_pipeline.pipeline.regression.specification import (
    RegressionSpecification,
    CovariateSpecification,
)


class RegressionDataInterface:

    def __init__(self,
                 fit_data_interface: FitDataInterface,
                 priors_root: Optional[io.CovariatePriorsRoot],
                 coefficient_root: Optional[io.RegressionRoot],
                 regression_root: io.RegressionRoot):
        self.fit_data_interface = fit_data_interface
        self.priors_root = priors_root
        self.coefficient_root = coefficient_root
        self.regression_root = regression_root

    @classmethod
    def from_specification(cls, specification: RegressionSpecification) -> 'RegressionDataInterface':
        fit_spec = FitSpecification.from_version_root(specification.data.seir_fit_version)
        fit_data_interface = FitDataInterface.from_specification(fit_spec)

        if specification.data.seir_covariate_priors_version:
            priors_root = io.CovariatePriorsRoot(specification.data.seir_covariate_priors_version)
        else:
            priors_root = None

        if specification.data.coefficient_version:
            coefficient_spec = RegressionSpecification.from_version_root(specification.data.coefficient_version)
            coefficient_root = io.RegressionRoot(coefficient_spec.data.output_root,
                                                 data_format=coefficient_spec.data.output_format)
        else:
            coefficient_root = None

        regression_root = io.RegressionRoot(specification.data.output_root,
                                            data_format=specification.data.output_format)

        return cls(
            fit_data_interface=fit_data_interface,
            priors_root=priors_root,
            coefficient_root=coefficient_root,
            regression_root=regression_root,
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.regression_root, **prefix_args)

    def get_n_draws(self) -> int:
        return self.fit_data_interface.get_n_draws()

    def filter_location_ids(self, hierarchy: pd.DataFrame = None) -> List[int]:
        """Get the list of location ids to model.

        This list is the intersection of a location metadata's
        locations, if provided, and the available locations in the infections
        directory.
        """
        regression_spec = self.load_specification()
        drop_locations = set(regression_spec.data.drop_locations)

        past_infections_locations = set(self.load_fit_location_ids())

        if hierarchy is None:
            desired_locations = past_infections_locations
        else:
            most_detailed = hierarchy.most_detailed == 1
            desired_locations = set(hierarchy.loc[most_detailed, 'location_id'].tolist())

        # past infections locations may be larger than desired locations, but we
        # do not care about the extras, only what's missing.
        past_infections_missing_locations = list(desired_locations - past_infections_locations)

        if past_infections_missing_locations:
            logger.warning("Some locations present in location metadata are missing from the "
                           f"infection models. Missing locations are "
                           f"{sorted(list(past_infections_missing_locations))}.")
        if drop_locations:
            logger.warning("Some locations present in the location metadata are being dropped. "
                           f"Locations being dropped are {sorted(list(drop_locations))}.")

        return list((past_infections_locations & desired_locations) - drop_locations)

    ####################
    # Prior Stage Data #
    ####################

    def load_hierarchy(self, name: str) -> pd.DataFrame:
        return self.fit_data_interface.load_hierarchy(name=name)

    def load_fit_location_ids(self) -> List[int]:
        return self.fit_data_interface.load_location_ids()

    def load_population(self, measure: str) -> pd.DataFrame:
        return self.fit_data_interface.load_population(measure=measure)

    def load_reported_epi_data(self) -> pd.DataFrame:
        return self.fit_data_interface.load_reported_epi_data()

    def load_hospital_census_data(self) -> pd.DataFrame:
        return self.fit_data_interface.load_hospital_census_data()

    def load_hospital_bed_capacity(self) -> pd.DataFrame:
        return self.fit_data_interface.load_hospital_bed_capacity()

    def load_total_covid_scalars(self, draw_id: int = None) -> pd.DataFrame:
        return self.fit_data_interface.load_total_covid_scalars(draw_id=draw_id)

    def load_seroprevalence(self, draw_id: int = None) -> pd.DataFrame:
        return self.fit_data_interface.load_seroprevalence(draw_id=draw_id)

    def load_sensitivity(self, draw_id: int = None) -> pd.DataFrame:
        return self.fit_data_interface.load_sensitivity(draw_id)

    def load_covariate(self,
                       covariate: str,
                       covariate_version: str = 'reference',
                       with_observed: bool = False) -> pd.DataFrame:
        covariate_df = self.fit_data_interface.load_covariate(covariate, covariate_version)
        location_ids = self.load_location_ids()
        covariate_df = self._format_covariate_data(covariate_df, location_ids, with_observed)
        covariate_df = (covariate_df
                        .rename(columns={f'{covariate}_{covariate_version}': covariate})
                        .loc[:, [covariate]])
        return covariate_df

    @staticmethod
    def _format_covariate_data(dataset: pd.DataFrame, location_ids: List[int], with_observed: bool = False):
        shared_locs = list(set(dataset.index.get_level_values('location_id')).intersection(location_ids))
        dataset = dataset.loc[shared_locs]
        if with_observed and 'observed' in dataset:
            dataset = dataset.set_index('observed', append=True)
        return dataset

    def load_covariates(self, covariates: Iterable[str]) -> pd.DataFrame:
        if not isinstance(covariates, dict):
            covariates = {c: 'reference' for c in covariates}
        covariate_data = []
        for covariate, covariate_version in covariates.items():
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate, covariate_version, with_observed=False))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data

    def load_covariate_info(self, covariate: str, info_type: str) -> pd.DataFrame:
        location_ids = self.load_location_ids()
        info_df = self.fit_data_interface.load_covariate_info(covariate, info_type)
        info_df = self._format_covariate_data(info_df, location_ids)
        return info_df

    def load_variant_prevalence(self, scenario: str) -> pd.DataFrame:
        return self.fit_data_interface.load_variant_prevalence(scenario)

    def load_waning_parameters(self, measure: str) -> pd.DataFrame:
        return self.fit_data_interface.load_waning_parameters(measure)

    def load_vaccine_summary(self, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_vaccine_summary(columns=columns)

    def load_vaccine_uptake(self, scenario: str) -> pd.DataFrame:
        return self.fit_data_interface.load_vaccine_uptake(scenario)

    def load_vaccine_risk_reduction(self, scenario: str) -> pd.DataFrame:
        return self.fit_data_interface.load_vaccine_risk_reduction(scenario)

    def load_covariate_options(self, draw_id: int = None) -> Dict:
        return self.fit_data_interface.load_covariate_options(draw_id)

    def load_ode_params(self, draw_id: int) -> pd.DataFrame:
        return self.fit_data_interface.load_ode_params(draw_id)

    def load_phis(self, draw_id: int) -> pd.DataFrame:
        return self.fit_data_interface.load_phis(draw_id)

    def load_input_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_input_epi_measures(draw_id, columns=columns)

    def load_rates_data(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_rates_data(draw_id, columns=columns)

    def load_rates(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_rates(draw_id, columns=columns)

    def load_antiviral_coverage(self, scenario: str) -> pd.DataFrame:
        return self.fit_data_interface.load_antiviral_coverage(scenario)

    def load_antiviral_effectiveness(self, draw_id: int) -> pd.DataFrame:
        return self.fit_data_interface.load_antiviral_effectiveness(draw_id=draw_id)

    def load_posterior_epi_measures(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_posterior_epi_measures(draw_id, columns=columns)

    def load_compartments(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_compartments(draw_id, columns=columns)

    def load_fit_beta(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_fit_beta(draw_id, columns=columns)

    def load_final_seroprevalence(self, draw_id: int, columns: List[str] = None) -> pd.DataFrame:
        return self.fit_data_interface.load_final_seroprevalence(draw_id, columns=columns)

    def load_summary(self, measure: str) -> pd.DataFrame:
        return self.fit_data_interface.load_summary(measure)

    ##########################
    # Covariate data loaders #
    ##########################

    def load_priors(self, covariates: Iterable[CovariateSpecification]) -> Dict[str, pd.DataFrame]:
        cov_names = [cov.name for cov in covariates if cov.gprior == 'data']
        if not cov_names:
            return {}

        if not self.priors_root:
            raise ValueError(f'Covariates {cov_names} specified data-based priors but no priors version '
                             f'was specified.')
        priors = io.load(self.priors_root.priors())
        missing_priors = set(cov_names).difference(priors.covariate.unique())
        if missing_priors:
            raise ValueError(f'Covariates {missing_priors} specified data-based priors but no priors were '
                             f'found in the specified covariate priors version.')
        out = {}
        for covariate in covariates:
            if covariate.gprior == 'data':
                covariate_prior = (
                    priors
                    .set_index(['covariate', covariate.group_level])
                    .loc[covariate.name, ['mean', 'sd']]
                    .drop_duplicates()
                )
                assert set(covariate_prior.index) == set(covariate_prior.index.drop_duplicates())
                out[covariate.name] = covariate_prior

        return out

    #######################
    # Regression data I/O #
    #######################

    def save_specification(self, specification: RegressionSpecification) -> None:
        io.dump(specification.to_dict(), self.regression_root.specification())

    def load_specification(self) -> RegressionSpecification:
        spec_dict = io.load(self.regression_root.specification())
        return RegressionSpecification.from_dict(spec_dict)

    def save_location_ids(self, location_ids: List[int]) -> None:
        io.dump(location_ids, self.regression_root.locations())

    def load_location_ids(self) -> List[int]:
        return io.load(self.regression_root.locations())

    def save_regression_beta(self, betas: pd.DataFrame, draw_id: int) -> None:
        io.dump(betas, self.regression_root.beta(draw_id=draw_id))

    def load_regression_beta(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.beta(draw_id=draw_id))

    def save_coefficients(self, coefficients: pd.DataFrame, draw_id: int) -> None:
        io.dump(coefficients, self.regression_root.coefficients(draw_id=draw_id))

    def load_coefficients(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.coefficients(draw_id=draw_id))

    def load_prior_run_coefficients(self, draw_id: int) -> Union[None, pd.DataFrame]:
        if self.coefficient_root:
            return io.load(self.coefficient_root.coefficients(draw_id=draw_id))
        return None

    def save_hospitalizations(self, data: pd.DataFrame, measure: str) -> None:
        io.dump(data, self.regression_root.hospitalizations(measure=measure))

    def load_hospitalizations(self, measure: str) -> pd.DataFrame:
        return io.load(self.regression_root.hospitalizations(measure=measure))

from functools import reduce
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Union

from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.regression.specification import (
    RegressionSpecification,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    HospitalFatalityRatioData,
    HospitalCensusData,
)


# TODO: move data interfaces up a package level and fuse with forecast data interface.

class RegressionDataInterface:

    def __init__(self,
                 infection_root: io.InfectionRoot,
                 covariate_root: io.CovariateRoot,
                 mortality_rate_root: io.MortalityRateRoot,
                 hospital_fatality_ratio_root: io.HospitalFatalityRatioRoot,
                 coefficient_root: Optional[io.RegressionRoot],
                 regression_root: io.RegressionRoot):
        self.infection_root = infection_root
        self.covariate_root = covariate_root
        self.mortality_rate_root = mortality_rate_root
        self.hospital_fatality_ratio_root = hospital_fatality_ratio_root
        self.coefficient_root = coefficient_root
        self.regression_root = regression_root

    @classmethod
    def from_specification(cls, specification: RegressionSpecification) -> 'RegressionDataInterface':
        # TODO: specify input format from config
        infection_root = io.InfectionRoot(specification.data.infection_version)
        covariate_root = io.CovariateRoot(specification.data.covariate_version)
        mortality_rate_root = io.MortalityRateRoot(specification.data.mortality_rate_version)
        hospital_fatality_ratio_root = io.HospitalFatalityRatioRoot(specification.data.hospital_fatality_ratio_version)
        if specification.data.coefficient_version:
            coefficient_root = io.RegressionRoot(specification.data.coefficient_version)
        else:
            coefficient_root = None
        # TODO: specify output format from config.
        regression_root = io.RegressionRoot(specification.data.output_root)

        return cls(
            infection_root=infection_root,
            covariate_root=covariate_root,
            mortality_rate_root=mortality_rate_root,
            hospital_fatality_ratio_root=hospital_fatality_ratio_root,
            coefficient_root=coefficient_root,
            regression_root=regression_root,
        )

    def make_dirs(self, **prefix_args) -> None:
        io.touch(self.regression_root, **prefix_args)

    def get_n_draws(self) -> int:
        regression_spec = io.load(self.regression_root.specification())
        return regression_spec['regression_parameters']['n_draws']

    #########################
    # Raw location handling #
    #########################

    def load_hierarchy_from_primary_source(self, location_set_version_id: Optional[int],
                                           location_file: Optional[Union[str, Path]]) -> Union[pd.DataFrame, None]:
        """Retrieve a location hierarchy from a file or from GBD if specified."""
        if location_set_version_id:
            location_metadata = utilities.load_location_hierarchy(location_set_id=111,
                                                                  location_set_version_id=location_set_version_id)
        elif location_file:
            location_metadata = utilities.load_location_hierarchy(location_file=location_file)
        else:
            location_metadata = None

        return location_metadata

    def filter_location_ids(self, desired_location_hierarchy: pd.DataFrame = None) -> List[int]:
        """Get the list of location ids to model.

        This list is the intersection of a location metadata's
        locations, if provided, and the available locations in the infections
        directory.

        """
        draw_0_data = self.load_full_past_infection_data(draw_id=0)
        total_deaths = draw_0_data.groupby('location_id').deaths.sum()
        modeled_locations = total_deaths[total_deaths > 5].index.tolist()

        if desired_location_hierarchy is None:
            desired_locations = modeled_locations
        else:
            most_detailed = desired_location_hierarchy.most_detailed == 1
            desired_locations = desired_location_hierarchy.loc[most_detailed, 'location_id'].tolist()

        missing_locations = list(set(desired_locations).difference(modeled_locations))
        if missing_locations:
            logger.warning("Some locations present in location metadata are missing from the "
                           f"infection models. Missing locations are {sorted(missing_locations)}.")
        return list(set(desired_locations).intersection(modeled_locations))

    ##########################
    # Infection data loaders #
    ##########################

    def load_full_past_infection_data(self, draw_id: int) -> pd.DataFrame:
        infection_data = io.load(self.infection_root.infections(draw_id=draw_id))
        infection_data['date'] = pd.to_datetime(infection_data['date'])
        infection_data = (infection_data
                          .set_index(['location_id', 'date'])
                          .sort_index()
                          .loc[:, ['infections_draw', 'deaths']]
                          .rename(columns={'infections_draw': 'infections'}))
        return infection_data

    def load_past_infection_data(self, draw_id: int):
        location_ids = self.load_location_ids()
        infection_data = self.load_full_past_infection_data(draw_id=draw_id)
        return infection_data.loc[location_ids]

    ##########################
    # Covariate data loaders #
    ##########################

    def check_covariates(self, covariates: Iterable[str]) -> None:
        """Ensure a reference scenario exists for all covariates.

        The reference scenario file is used to find the covariate values
        in the past (which we'll use to perform the regression).

        """
        missing = []

        for covariate in covariates:
            if covariate != 'intercept':
                if not io.exists(self.covariate_root[covariate](covariate_scenario='reference')):
                    missing.append(covariate)

        if missing:
            raise ValueError('All covariates supplied in the regression specification'
                             'must have a reference scenario in the covariate pool. Covariates'
                             f'missing a reference scenario: {missing}.')

    def load_covariate(self, covariate: str) -> pd.DataFrame:
        location_ids = self.load_location_ids()
        covariate_df = io.load(self.covariate_root[covariate](covariate_scenario='reference'))
        index_columns = ['location_id']
        covariate_df = covariate_df.loc[covariate_df['location_id'].isin(location_ids), :]
        if 'date' in covariate_df.columns:
            covariate_df['date'] = pd.to_datetime(covariate_df['date'])
            index_columns.append('date')
        covariate_df = covariate_df.rename(columns={f'{covariate}_reference': covariate})
        return covariate_df.loc[:, index_columns + [covariate]].set_index(index_columns)

    def load_covariates(self, covariates: Iterable[str]) -> pd.DataFrame:
        covariate_data = []
        for covariate in covariates:
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data.reset_index()

    ######################
    # Ratio data loaders #
    ######################

    def load_ifr_data(self, draw_id: int, location_ids: List[int]) -> pd.DataFrame:
        ifr = io.load(self.infection_root.ifr(draw_id=draw_id))
        ifr = ifr[ifr.location_id.isin(location_ids)]
        ifr['date'] = pd.to_datetime(ifr['date'])
        ifr = ifr.set_index(['location_id', 'date', 'duration']).sort_index()
        cols = [c for c in ifr.columns if '_draw' in c]
        ifr = ifr.loc[:, cols].rename(columns={c: c.split('_draw')[0] for c in cols})
        return ifr.reset_index(level='duration')

    def load_mortality_ratio(self) -> pd.Series:
        location_ids = self.load_location_ids()
        mr_df = io.load(self.mortality_rate_root.mortality_rate())
        mr = mr_df.set_index(['location_id', 'age_start']).MRprob
        return mr.loc[location_ids]

    def load_hospital_fatality_ratio(self,
                                     death_weights: pd.Series,
                                     with_error: bool) -> 'HospitalFatalityRatioData':
        location_ids = self.load_location_ids()
        hfr_age_cols = ['X1', 'X2', 'X3', 'X4', 'X5']

        hfr_all_locs = io.load(self.hospital_fatality_ratio_root.hospital_fatality_ratio())
        hfr = (hfr_all_locs[hfr_all_locs.location_id.isin(location_ids)]
               .drop(columns='location')
               .set_index('location_id'))

        # TODO: Why round?  Why mode?
        # For missing locations, use the rounded mode of the all loc hfr to fill.
        missing_locs = set(location_ids).difference(hfr.index)
        if missing_locs:
            if with_error:
                fill_hfr = hfr_all_locs[hfr_age_cols + ['all_age']].round().mode()
            else:
                fill_hfr = hfr_all_locs[hfr_age_cols + ['all_age']].mean().to_frame().T
            missing_hfr = pd.concat([fill_hfr] * len(missing_locs))
            missing_hfr.index = pd.Index(missing_locs, name='location_id')
            hfr = hfr.append(missing_hfr).sort_index()

        # TODO: Why are we rounding?  Mysterious...
        if with_error:
            hfr_all_age = hfr['all_age'].round()
            hfr = hfr[hfr_age_cols].round()
        else:
            hfr_all_age = hfr['all_age'].round()
            hfr = hfr[hfr_age_cols].round()
        actual_ages = sorted(death_weights.reset_index().age.unique())
        assert len(hfr_age_cols) == len(actual_ages), 'Something terrible has happened to the hfr age pattern.'
        hfr.columns = actual_ages

        low_hfr = hfr[(hfr < 1).any(axis=1)].index.tolist()
        if low_hfr:
            logger.warning(f'HDR below 1 found in locations {low_hfr}')
        hfr = hfr.clip(1).stack()
        hfr.index.names = ['location_id', 'age']
        hfr.name = None
        return HospitalFatalityRatioData(age_specific=hfr, all_age=hfr_all_age)

    ##############################
    # Miscellaneous data loaders #
    ##############################

    def get_infectionator_metadata(self):
        return io.load(self.infection_root.metadata())

    def get_model_inputs_metadata(self):
        infection_metadata = self.get_infectionator_metadata()
        return infection_metadata['model_inputs_metadata']

    def load_population(self) -> pd.DataFrame:
        metadata = self.get_model_inputs_metadata()
        model_inputs_version = metadata['output_path']
        population_path = Path(model_inputs_version) / 'output_measures' / 'population' / 'all_populations.csv'
        population_data = pd.read_csv(population_path)
        return population_data

    def load_five_year_population(self) -> pd.DataFrame:
        population = self.load_population()
        location_ids = self.load_location_ids()
        in_locations = population['location_id'].isin(location_ids)
        is_2019 = population['year_id'] == 2019
        is_both_sexes = population['sex_id'] == 3
        five_year_bins = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
        is_five_year_bins = population['age_group_id'].isin(five_year_bins)
        population = population.loc[in_locations & is_2019 & is_both_sexes & is_five_year_bins, :]
        return population

    def load_hospital_census_data(self) -> 'HospitalCensusData':
        metadata = self.get_model_inputs_metadata()

        model_inputs_path = Path(metadata['output_path'])
        corrections_data = {}
        file_map = (
            ('hospitalizations', 'hospital_census'),
            ('icu', 'icu_census'),
            ('ventilators', 'ventilator_census'),
        )
        for dir_name, measure in file_map:
            path = model_inputs_path / 'output_measures' / dir_name / 'population.csv'
            df = pd.read_csv(path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.loc[(df.age_group_id == 22) & (df.sex_id == 3)]
            df = df[["location_id", "date", "value"]]
            if df.groupby(["location_id", "date"]).count().value.max() > 1:
                raise ValueError(f"Duplicate usages for location_id and date in {path}")
            # Location IDs to exclude from the census input data. Requested by Steve on 9/30
            census_exclude_locs = [200, 69, 179, 172, 170, 144, 26, 74, 67, 58]
            df = df.loc[~df.location_id.isin(census_exclude_locs)]
            corrections_data[measure] = df.set_index(['location_id', 'date']).value
        return HospitalCensusData(**corrections_data)

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

    def save_hierarchy(self, hierarchy: pd.DataFrame) -> None:
        io.dump(hierarchy, self.regression_root.hierarchy())

    def load_hierarchy(self) -> pd.DataFrame:
        return io.load(self.regression_root.hierarchy())

    def save_beta_param_file(self, df: pd.DataFrame, draw_id: int) -> None:
        io.dump(df, self.regression_root.parameters(draw_id=draw_id))

    def load_beta_param_file(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.parameters(draw_id=draw_id))

    def save_date_file(self, df: pd.DataFrame, draw_id: int) -> None:
        io.dump(df, self.regression_root.dates(draw_id=draw_id))

    def load_date_file(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.dates(draw_id=draw_id))

    def save_regression_coefficients(self, coefficients: pd.DataFrame, draw_id: int) -> None:
        io.dump(coefficients, self.regression_root.coefficients(draw_id=draw_id))

    def load_regression_coefficients(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.coefficients(draw_id=draw_id))

    def load_prior_run_coefficients(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.coefficient_root.coefficients(draw_id=draw_id))

    def save_regression_betas(self, df: pd.DataFrame, draw_id: int) -> None:
        io.dump(df, self.regression_root.beta(draw_id=draw_id))

    def load_regression_betas(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.beta(draw_id=draw_id))

    def save_infection_data(self, df: pd.DataFrame, draw_id: int) -> None:
        io.dump(df, self.regression_root.infection_data(draw_id=draw_id))

    def load_infection_data(self, draw_id: int) -> pd.DataFrame:
        return io.load(self.regression_root.infection_data(draw_id=draw_id))

    def save_hospital_data(self, df: pd.DataFrame, measure: str) -> None:
        io.dump(df, self.regression_root.hospitalizations(measure=measure))

    def load_hospital_data(self, measure: str) -> pd.DataFrame:
        return io.load(self.regression_root.hospitalizations(measure=measure))

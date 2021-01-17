from functools import reduce
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Union

from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    io,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification


# TODO: move data interfaces up a package level and fuse with forecast data interface.

class RegressionDataInterface:

    def __init__(self,
                 infection_root: io.InfectionRoot,
                 covariate_root: io.CovariateRoot,
                 coefficient_root: Optional[io.RegressionRoot],
                 regression_root: io.RegressionRoot):
        self.infection_root = infection_root
        self.covariate_root = covariate_root
        self.coefficient_root = coefficient_root
        self.regression_root = regression_root

    @classmethod
    def from_specification(cls, specification: RegressionSpecification) -> 'RegressionDataInterface':
        # TODO: specify input format from config
        infection_root = io.InfectionRoot(specification.data.infection_version)
        covariate_root = io.CovariateRoot(specification.data.covariate_version)
        if specification.data.coefficient_version:
            coefficient_root = io.RegressionRoot(specification.data.coefficient_version)
        else:
            coefficient_root = None
        # TODO: specify output format from config.
        regression_root = io.RegressionRoot(specification.data.output_root)

        return cls(
            infection_root=infection_root,
            covariate_root=covariate_root,
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
        modeled_locations = self.infection_root.modeled_locations()
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

    def load_all_location_data(self, location_ids: List[int], draw_id: int) -> Dict[int, pd.DataFrame]:
        dfs = dict()
        for loc in location_ids:
            loc_df = io.load(self.infection_root.infections(location_id=loc, draw_id=draw_id))
            loc_df = loc_df.rename(columns={'loc_id': 'location_id'})
            if loc_df['cases_draw'].isnull().any():
                logger.warning(f'Nulls found in infectionator inputs for location id {loc}.  Dropping.')
                continue
            if (loc_df['cases_draw'] < 0).any():
                logger.warning(f'Negatives found in infectionator inputs for location id {loc}.  Dropping.')
                continue
            dfs[loc] = loc_df

        return dfs

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

    def load_covariate(self, covariate: str, location_ids: List[int]) -> pd.DataFrame:
        covariate_df = io.load(self.covariate_root[covariate](covariate_scenario='reference'))
        index_columns = ['location_id']
        covariate_df = covariate_df.loc[covariate_df['location_id'].isin(location_ids), :]
        if 'date' in covariate_df.columns:
            covariate_df['date'] = pd.to_datetime(covariate_df['date'])
            index_columns.append('date')
        covariate_df = covariate_df.rename(columns={f'{covariate}_reference': covariate})
        return covariate_df.loc[:, index_columns + [covariate]].set_index(index_columns)

    def load_covariates(self, covariates: Iterable[str], location_ids: List[int]) -> pd.DataFrame:
        covariate_data = []
        for covariate in covariates:
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate, location_ids))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data.reset_index()

    ######################
    # Ratio data loaders #
    ######################

    def load_ifr_data(self):
        metadata = self.get_infectionator_metadata()
        # TODO: metadata abstraction?
        ifr_version = metadata['run_arguments']['ifr_custom_path']
        data_path = Path(ifr_version) / 'terminal_ifr.csv'
        data = pd.read_csv(data_path)
        return data.set_index('location_id')

    ##############################
    # Miscellaneous data loaders #
    ##############################

    def get_infectionator_metadata(self):
        return io.load(self.infection_root.metadata())

    def get_model_inputs_metadata(self):
        infection_metadata = self.get_infectionator_metadata()
        return infection_metadata['death']['metadata']['model_inputs_metadata']

    def load_population(self) -> pd.DataFrame:
        metadata = self.get_model_inputs_metadata()
        model_inputs_version = metadata['output_path']
        population_path = Path(model_inputs_version) / 'output_measures' / 'population' / 'all_populations.csv'
        population_data = pd.read_csv(population_path)
        return population_data

    def load_five_year_population(self, location_ids: List[int]) -> pd.DataFrame:
        population = self.load_population()
        in_locations = population['location_id'].isin(location_ids)
        is_2019 = population['year_id'] == 2019
        is_both_sexes = population['sex_id'] == 3
        five_year_bins = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
        is_five_year_bins = population['age_group_id'].isin(five_year_bins)
        population = population.loc[in_locations & is_2019 & is_both_sexes & is_five_year_bins, :]
        return population

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

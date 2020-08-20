from functools import reduce
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Union

from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
    Keys as MKeys,
)
from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification


# TODO: move data interfaces up a package level and fuse with forecast data interface.

class RegressionDataInterface:

    def __init__(self,
                 regression_paths: paths.RegressionPaths,
                 infection_paths: paths.InfectionPaths,
                 covariate_paths: paths.CovariatePaths,
                 regression_marshall,
                 ):
        # TODO: only hang on to marshalls here.
        self.regression_paths = regression_paths
        self.infection_paths = infection_paths
        self.covariate_paths = covariate_paths
        self.regression_marshall = regression_marshall

    @classmethod
    def from_specification(cls, specification: RegressionSpecification) -> 'RegressionDataInterface':
        regression_paths = paths.RegressionPaths(Path(specification.data.output_root), read_only=False)
        infection_paths = paths.InfectionPaths(Path(specification.data.infection_version))
        covariate_paths = paths.CovariatePaths(Path(specification.data.covariate_version))
        # TODO: specification of marshall type from inference on inputs and
        #   configuration on outputs.
        return cls(
            regression_paths=regression_paths,
            infection_paths=infection_paths,
            covariate_paths=covariate_paths,
            regression_marshall=CSVMarshall.from_paths(regression_paths),
        )

    def make_dirs(self):
        self.regression_paths.make_dirs()

    #####################
    # Location handling #
    #####################

    def load_location_ids_from_primary_source(self, location_set_version_id: Optional[int],
                                              location_file: Optional[Union[str, Path]]) -> Union[List[int], None]:
        """Retrieve a location hierarchy from a file or from GBD if specified."""
        # TODO: Remove after integration testing.
        assert not (location_set_version_id and location_file), 'CLI location validation is broken.'
        if location_set_version_id:
            location_metadata = self._load_from_location_set_version_id(location_set_version_id)
        elif location_file:
            location_metadata = pd.read_csv(location_file)
        else:
            location_metadata = None

        if location_metadata is not None:
            location_metadata = location_metadata.loc[location_metadata.most_detailed == 1, 'location_id'].tolist()
        return location_metadata

    def filter_location_ids(self, desired_locations: List[int] = None) -> List[int]:
        """Get the list of location ids to model.

        This list is the intersection of a location metadata's
        locations, if provided, and the available locations in the infections
        directory.

        """
        modeled_locations = self.infection_paths.get_modelled_locations()
        if desired_locations is None:
            desired_locations = modeled_locations

        missing_locations = list(set(desired_locations).difference(modeled_locations))
        if missing_locations:
            logger.warning("Some locations present in location metadata are missing from the "
                           f"infection models. Missing locations are {sorted(missing_locations)}.")
        return list(set(desired_locations).intersection(modeled_locations))

    def load_location_ids(self) -> List[int]:
        with self.regression_paths.location_metadata.open() as location_file:
            location_ids = yaml.full_load(location_file)
        return location_ids

    def dump_location_ids(self, location_ids: List[int]):
        with self.regression_paths.location_metadata.open('w') as location_file:
            yaml.dump(location_ids, location_file)

    ##########################
    # Infection data loaders #
    ##########################

    def load_all_location_data(self, location_ids: List[int], draw_id: int
                               ) -> Dict[int, pd.DataFrame]:
        dfs = dict()
        for loc in location_ids:
            file = self.infection_paths.get_infection_file(location_id=loc, draw_id=draw_id)
            loc_df = pd.read_csv(file).rename(columns={'loc_id': 'location_id'})
            if loc_df['cases_draw'].isnull().any():
                logger.warning(f'Nulls found in infectionator inputs for location id {loc}.  Dropping.')
                continue
            if (loc_df['cases_draw'] < 0).any():
                logger.warning(f'Negatives found in infectionator inputs for location id {loc}.  Dropping.')
                continue
            dfs[loc] = loc_df

        return dfs

    ###########################
    # Covariate paths loaders #
    ###########################

    def check_covariates(self, covariates: Iterable[str]):
        """Ensure a reference scenario exists for all covariates.

        The reference scenario file is used to find the covariate values
        in the past (which we'll use to perform the regression).

        """
        missing = []

        for covariate in covariates:
            if covariate != 'intercept':
                covariate_path = self.covariate_paths.get_covariate_scenario_file(covariate, 'reference')
                if not covariate_path.exists():
                    missing.append(covariate)

        if missing:
            raise ValueError('All covariates supplied in the regression specification'
                             'must have a reference scenario in the covariate pool. Covariates'
                             f'missing a reference scenario: {missing}.')

    def load_covariate(self, covariate: str, location_ids: List[int]) -> pd.DataFrame:
        covariate_path = self.covariate_paths.get_covariate_scenario_file(covariate, 'reference')
        covariate_df = pd.read_csv(covariate_path)
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

    ############################
    # Regression paths writers #
    ############################

    def save_beta_param_file(self, df: pd.DataFrame, draw_id: int) -> None:
        self.regression_marshall.dump(df, key=MKeys.parameter(draw_id))

    def save_date_file(self, df: pd.DataFrame, draw_id: int) -> None:
        self.regression_marshall.dump(df, key=MKeys.date(draw_id))

    def save_location_metadata_file(self, locations: List[int]) -> None:
        with (self.regression_paths.root_dir / 'locations.yaml') as location_file:
            yaml.dump({'locations': locations}, location_file)

    def save_regression_coefficients(self, coefficients: pd.DataFrame, draw_id: int) -> None:
        self.regression_marshall.dump(coefficients, key=MKeys.coefficient(draw_id))

    def save_regression_betas(self, df: pd.DataFrame, draw_id: int) -> None:
        self.regression_marshall.dump(df, key=MKeys.regression_beta(draw_id))

    def save_location_data(self, df: pd.DataFrame, draw_id: int) -> None:
        # quasi-inverse of load_all_location_data, except types are different
        self.regression_marshall.dump(df, key=MKeys.location_data(draw_id))

    @staticmethod
    def _load_from_location_set_version_id(location_set_version_id: int) -> pd.DataFrame:
        # Hide this import so the code stays portable outside IHME by using
        # a locations file directly.
        from db_queries import get_location_metadata
        return get_location_metadata(location_set_id=111,
                                     location_set_version_id=location_set_version_id)

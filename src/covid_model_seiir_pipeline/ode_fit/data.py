from pathlib import Path
from typing import List, Dict, Optional, Union

from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.static_vars import INFECTION_COL_DICT

class ODEDataInterface:

    def __init__(self, ode_paths: paths.ODEPaths,
                 infection_paths: paths.InfectionPaths) -> None:
        self.ode_paths = ode_paths
        self.infection_paths = infection_paths

    @classmethod
    def from_fit_specification(cls, fit_specification: FitSpecification):
        ode_paths = paths.ODEPaths(Path(fit_specification.data.output_root), read_only=False)
        infection_paths = paths.InfectionPaths(Path(fit_specification.data.infection_version))
        return cls(
            ode_paths=ode_paths,
            infection_paths=infection_paths,
        )

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
        with self.ode_paths.location_metadata.open() as location_file:
            location_ids = yaml.full_load(location_file)
        return location_ids

    def dump_location_ids(self, location_ids: List[int]):
        with self.ode_paths.location_metadata.open('w') as location_file:
            yaml.dump(location_ids, location_file)

    def load_all_location_data(self, location_ids: List[int], draw_id: int
                               ) -> Dict[int, pd.DataFrame]:
        dfs = dict()
        for loc in location_ids:
            file = self.infection_paths.get_infection_file(location_id=loc, draw_id=draw_id)
            dfs[loc] = pd.read_csv(file)

        # validate
        locs_na = []
        locs_neg = []
        for loc, df in dfs.items():
            if df[INFECTION_COL_DICT['COL_CASES']].isna().any():
                locs_na.append(loc)
            if (df[INFECTION_COL_DICT['COL_CASES']].to_numpy() < 0.0).any():
                locs_neg.append(loc)
        if len(locs_na) > 0 and len(locs_neg) > 0:
            raise ValueError(
                'NaN in infection data: ' + str(locs_na) + '. Negatives in infection data: ' +
                str(locs_neg)
            )
        if len(locs_na) > 0:
            raise ValueError('NaN in infection data: ' + str(locs_na))
        if len(locs_neg) > 0:
            raise ValueError('Negatives in infection data:' + str(locs_neg))

        return dfs

    def save_draw_beta_fit_file(self, df: pd.DataFrame, location_id: int, draw_id: int
                                ) -> None:
        df.to_csv(self.ode_paths.get_draw_beta_fit_file(location_id, draw_id), index=False)

    def save_draw_beta_param_file(self, df: pd.DataFrame, draw_id: int) -> None:
        df.to_csv(self.ode_paths.get_draw_beta_param_file(draw_id), index=False)

    def save_draw_date_file(self, df: pd.DataFrame, draw_id: int) -> None:
        df.to_csv(self.ode_paths.get_draw_date_file(draw_id), index=False)

    def save_location_metadata_file(self, locations: List[int]) -> None:
        with (self.ode_paths.root_dir / 'locations.yaml') as location_file:
            yaml.dump({'locations': locations},location_file)

    @staticmethod
    def _load_from_location_set_version_id(location_set_version_id: int) -> pd.DataFrame:
        # Hide this import so the code stays portable outside IHME by using
        # a locations file directly.
        from db_queries import get_location_metadata
        return get_location_metadata(location_set_id=111,
                                     location_set_version_id=location_set_version_id)

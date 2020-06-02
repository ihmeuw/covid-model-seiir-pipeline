from pathlib import Path
from typing import List, Dict

from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline.static_vars import INFECTION_COL_DICT
from covid_model_seiir_pipeline.paths import ODEPaths, InfectionPaths


class ODEDataInterface:

    def __init__(self, ode_fit_root: Path, infection_root: Path, location_file: Path) -> None:
        self.ode_paths = ODEPaths(ode_fit_root)
        self.infection_paths = InfectionPaths(infection_root)
        self.location_metadata_file = location_file

    def load_location_ids(self) -> List[int]:
        """Get the list of location ids to model.

        This list is the intersection of a location hierarchy file and
        the available locations in the infections directory.

        """
        desired_locs = pd.read_csv(self.location_metadata_file)["location_id"].tolist()
        modeled_locs = self.infection_paths.get_modelled_locations()
        missing_locs = list(set(desired_locs).difference(modeled_locs))
        if missing_locs:
            logger.warning("Some locations present in location metadata are missing from the "
                           f"infection models. Missing locations are {missing_locs}.")
        return list(set(desired_locs).intersection(modeled_locs))

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


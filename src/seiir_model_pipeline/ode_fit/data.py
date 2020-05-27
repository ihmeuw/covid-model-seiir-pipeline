from pathlib import Path
from typing import List, Dict

import pandas as pd


from seiir_model_pipeline.static_vars import INFECTION_COL_DICT
from seiir_model_pipeline.paths import ODEPaths, InfectionPaths
from seiir_model_pipeline.ode_fit.specification import FitData


class ODEDataInterface:

    def __init__(self, ode_fit_data: FitData) -> None:
        self.ode_paths = ODEPaths(Path(ode_fit_data.output_root))
        self.infection_paths = InfectionPaths(Path(ode_fit_data.infection_version))

        # TODO: figure out where this comes from
        file = f'location_metadata_{ode_fit_data.location_set_version_id}.csv'
        file_path = Path('/ihme/covid-19/seir-pipeline-outputs/metadata-inputs') / file
        self.location_metadata_file = file_path

    def load_location_ids(self) -> List[int]:
        return pd.read_csv(self.location_metadata_file)["location_id"].tolist()

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
        df.to_csv(self.ode_paths.get_draw_beta_param_file(draw_id))

    def save_draw_date_file(self, df: pd.DataFrame, draw_id: int) -> None:
        df.to_csv(self.ode_paths.get_draw_date_file(draw_id), index=False)

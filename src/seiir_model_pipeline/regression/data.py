from pathlib import Path
from typing import List
import warnings

import pandas as pd


from seiir_model_pipeline.regression.globals import (COVARIATE_COL_DICT, COVARIATES_FILE,
                                                     COVARIATE_DRAW_FILE)


class RegressionData:

    def __init__(self, regression_dir: Path):
        self.regression_dir = regression_dir

    @property
    def regression_beta_fit_dir(self) -> Path:
        return self.regression_dir / 'betas'

    @property
    def regression_parameters_dir(self) -> Path:
        return self.regression_dir / 'parameters'

    @property
    def regression_coefficient_dir(self) -> Path:
        return self.regression_dir / 'coefficients'

    @property
    def regression_diagnostic_dir(self) -> Path:
        return self.regression_dir / 'diagnostics'

    @property
    def regression_input_dir(self) -> Path:
        return self.regression_dir / 'inputs'

    @property
    def regression_covariate_dir(self) -> Path:
        return self.regression_dir / 'inputs' / 'covariates'

    def make_dirs(self) -> None:
        for directory in [self.regression_beta_fit_dir, self.regression_parameters_dir,
                          self.regression_coefficient_dir, self.regression_diagnostic_dir,
                          self.regression_input_dir, self.regression_covariates_dir]:
            directory.mkdir(mode=755, parents=True, exist_ok=True)

    def get_draw_coefficient_file(self, draw_id: int) -> Path:
        return self.regression_coefficient_dir / f'coefficients_{draw_id}.csv'

    def get_draw_beta_param_file(self, draw_id: int) -> Path:
        return self.regression_parameters_dir / f'params_draw_{draw_id}.csv'

    def get_draw_beta_fit_file(self, location_id: int, draw_id: int) -> Path:
        loc_dir = self.regression_beta_fit_dir / str(location_id)
        loc_dir.mkdir(mode=755, parents=True, exist_ok=True)
        return loc_dir / f'fit_draw_{draw_id}.csv'

    @property
    def location_ids(self) -> List[int]:
        loc_file = pd.read_csv(self.regression_input_dir / "locations.csv")
        return loc_file[["location_id"]].tolist()

    def load_covariates(self, draw_id=None):
        """
        Load covariates that have *already been cached*.

        :param draw_id:
        :return:
        """
        if draw_id is None:
            file = COVARIATES_FILE
        else:
            file = COVARIATE_DRAW_FILE.format(draw_id=draw_id)
        df = pd.read_csv(self.regression_covariate_dir / file)
        df = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']].isin(self.location_ids)].copy()
        return df

    def load_mr_coefficients(self, draw_id: int):
        """
        Load meta-regression coefficients

        :param directories: Directories object
        :param draw_id: (int) which draw to load
        :return:
        """
        df = pd.read_csv(self.get_draw_coefficient_file(draw_id))
        return df


class InfectionData:
    INFECTION_FILE_PATTERN = 'draw{draw_id:04}_prepped_deaths_and_cases_all_age.csv'

    def __init__(self, infection_dir: Path):
        self.infection_dir: infection_dir

    def get_location_dir(self, location_id: int):
        matches = self.infection_dir.glob(f"*_{location_id}")
        num_matches = len(matches)
        if num_matches > 1:
            raise RuntimeError("There is more than one location-specific folder for "
                               f"{location_id}.")
        elif num_matches == 0:
            raise FileNotFoundError("There is not a location-specific folder for "
                                    f"{location_id}.")
        else:
            folder = matches[0]
        return self.infection_dir / folder

    def get_infection_file(self, location_id: int, draw_id: int) -> Path:
        # folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        f = (self.infection_dir.get_location_dir(location_id) /
             self.INFECTION_FILE_PATTERN.format(draw_id=draw_id))
        return f

    def get_missing_infection_locations(self, location_ids: List[int]) -> List[int]:
        infection_loc = [str(d).split('_')[-1] for d in self.infection_dir.iterdir()
                         if d.is_dir()]
        infection_loc = [int(x) for x in infection_loc if x.isdigit()]
        missing_infection_loc = set(location_ids) - set(infection_loc)
        warnings.warn('Locations missing from infection data: ' + str(missing_infection_loc))
        return list(missing_infection_loc)

    def load_all_location_data(self, location_ids: List[int], draw_id: int):
        dfs = dict()
        for loc in location_ids:
            file = self.get_infection_file(location_id=loc, draw_id=draw_id)
            dfs[loc] = pd.read_csv(file)
        return dfs

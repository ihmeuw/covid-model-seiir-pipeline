from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os

BASE_DIR = Path('/ihme')

# Dependency directories
INPUT_DIR = BASE_DIR / 'covid-19/seir-inputs'
COVARIATE_DIR = BASE_DIR / 'fake/covariate/input/dir'

# Output directories
DIAGNOSTIC_DIR = BASE_DIR / 'fake/diagnostics/dir'
DRAW_DIR = BASE_DIR / 'fake/draw/dir'
OUTPUT_DIR = BASE_DIR / 'fake/output/dir'

LOG_DIR = BASE_DIR / 'fake/log/dir'

INFECTION_FILE_PATTERN = 'draw{draw_id}_prepped_deaths_and_cases_all_age.csv'
PEAK_DATE_FILE = '/ihme/scratch/projects/covid/seir_research_test_run/death_model_peaks.csv'
PEAK_DATE_COL_DICT = {
    'COL_LOC_ID': 'location_id',
    'COL_DATE': 'peaked_date'
}

INFECTION_COL_DICT = {
    'COL_DATE': 'date',
    'COL_CASES': 'cases',
    'COL_POP': 'pop',
    'COL_LOC_ID': 'loc_id'
}


def _get_infection_folder_from_location_id(location_id, input_dir):
    """
    This is the infection input folder. It's a helper because
    folders have location names in them.

    :param location_id: (int)
    :param input_dir: (Path)
    :return: (str)
    """
    folders = os.listdir(input_dir)
    correct = np.array([f.endswith(f'_{location_id}') for f in folders])
    if correct.sum() > 1:
        raise RuntimeError(f"There is more than one location-specific folder for {location_id}.")
    elif correct.sum() == 0:
        raise RuntimeError(f"There is not a location-specific folder for {location_id}.")
    else:
        pass
    folder = folders[np.where(correct)[0][0]]
    return folder


def _get_loc_scenario_draw_file(location_id, draw_id, scenario_id, directory):
    """
    This is the location-scenario-draw file.

    :param location_id: (int)
    :param draw_id: (int)
    :param scenario_id: (int)
    :param directory: (Path) parent directory
    :return: (Path)
    """
    return directory / f'{location_id}/draw{draw_id}_scenario{scenario_id}.csv'


def _get_loc_scenario_file(location_id, scenario_id, directory):
    """
    This is the final location-scenario file with all draws.

    :param location_id: (int)
    :param scenario_id: (int)
    :param directory: (Path) parent directory
    :return: (Path)
    """
    return directory / f'{location_id}/scenario{scenario_id}.csv'


def args_to_directories(args):
    """

    :param args: result of an argparse.ArgumentParser.parse_args()
    :return: (Directories) object
    """
    return Directories(
        infection_version=args.infection_version,
        covariate_version=args.covariate_version,
        output_version=args.output_version
    )


@dataclass
class Directories:
    """
    ## Arguments

    - `infection_version (str)`: version of the infections to pull
    - `covariate_version (str)`: version of the covariates to pull
    - `output_version (str)`: version of outputs to store
    """

    infection_version: str
    covariate_version: str
    output_version: str

    def __post_init__(self):
        self.infection_dir = INPUT_DIR / self.infection_version
        self.covariate_dir = BASE_DIR / self.covariate_version
        self.output_dir = BASE_DIR / self.output_dir
        self.log_dir = LOG_DIR / self.output_version

        self.draw_dir = self.output_dir / 'draws'
        self.scenario_draw_dir = self.draw_dir / 'location_draws'
        self.scenario_dir = self.draw_dir / 'location_scenario_draws'
        self.diagnostic_dir = self.output_dir / 'diagnostics'

        self.error_dir = self.log_dir / 'errors'
        self.output_dir = self.log_dir / 'output'

        self.settings_file = self.output_dir / 'settings.json'

    def make_dirs(self):
        for directory in [
            self.draw_dir, self.scenario_draw_dir,
            self.scenario_dir, self.diagnostic_dir,
            self.error_dir, self.output_dir
        ]:
            os.makedirs(directory, exist_ok=True)

    def make_location_directories(self, location_id):
        for directory in [
            self.scenario_draw_dir, self.scenario_dir, self.diagnostic_dir
        ]:
            os.makedirs(directory / str(location_id), exist_ok=True)

    def get_infection_file(self, location_id, draw_id):
        folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        return self.infection_dir / folder / INFECTION_FILE_PATTERN.format(draw_id=draw_id)

    def get_draw_scenario_output_file(self, location_id, draw_id, scenario_id):
        return _get_loc_scenario_draw_file(
            location_id=location_id, draw_id=draw_id, scenario_id=scenario_id,
            directory=self.scenario_draw_dir
        )

    def get_scenario_output_file(self, location_id, scenario_id):
        return _get_loc_scenario_file(
            location_id=location_id, scenario_id=scenario_id,
            directory=self.scenario_dir
        )

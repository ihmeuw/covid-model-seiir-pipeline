from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os

from seiir_model_pipeline.core.versioner import load_forecast_settings, load_regression_settings

BASE_DIR = Path('/ihme')

# Dependency directories
INPUT_DIR = BASE_DIR / 'covid-19/seir-inputs'
COVARIATE_DIR = BASE_DIR / 'fake/covariate/input/dir'

# Output directories
DIAGNOSTIC_DIR = BASE_DIR / 'fake/diagnostics/dir'
DRAW_DIR = BASE_DIR / 'fake/draw/dir'
OUTPUT_DIR = BASE_DIR / 'fake/output/dir'

REGRESSION_OUTPUT = OUTPUT_DIR / 'regression'
FORECAST_OUTPUT = OUTPUT_DIR / 'forecast'

LOG_DIR = BASE_DIR / 'fake/log/dir'

INFECTION_FILE_PATTERN = 'draw{draw_id}_prepped_deaths_and_cases_all_age.csv'
PEAK_DATE_FILE = '/ihme/scratch/projects/covid/seir_research_test_run/death_model_peaks.csv'
COVARIATE_FILE = 'fake_inputs.csv'

INFECTION_COL_DICT = {
    'COL_DATE': 'date',
    'COL_CASES': 'cases',
    'COL_POP': 'pop',
    'COL_LOC_ID': 'loc_id'
}


def _get_regression_settings_file(regression_version):
    return REGRESSION_OUTPUT / str(regression_version) / 'settings.json'


def _get_forecast_settings_file(forecast_version):
    return FORECAST_OUTPUT / str(forecast_version) / 'settings.json'


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
        regression_version=args.regression_version,
        forecast_version=args.forecast_version
    )


@dataclass
class Directories:
    """
    ## Arguments

    - `infection_version (str)`: version of the infections to pull
    - `covariate_version (str)`: version of the covariates to pull
    - `output_version (str)`: version of outputs to store
    """

    regression_version: str = None
    forecast_version: str = None

    def __post_init__(self):
        rv = None
        fv = None

        if self.regression_version is None:
            if self.forecast_version is None:
                pass
            else:
                fv = load_forecast_settings(self.forecast_version)
                rv = load_regression_settings(fv.regression_version)
        else:
            rv = load_regression_settings(self.regression_version)
            if self.forecast_version is not None:
                fv = load_forecast_settings(self.forecast_version)

        if rv is not None:
            self.infection_dir = INPUT_DIR / rv.infection_version
            self.covariate_dir = BASE_DIR / rv.covariate_version

            self.regression_output_dir = BASE_DIR / REGRESSION_OUTPUT / rv.version_name

            self.regression_coefficient_dir = self.regression_output_dir / 'coefficients'
            self.regression_diagnostic_dir = self.regression_output_dir / 'diagnostics'

        if fv is not None:
            self.forecast_output_dir = BASE_DIR / FORECAST_OUTPUT / fv.version_name

            self.forecast_draw_dir = self.forecast_output_dir / 'location_draws'
            self.forecast_diagnostic_dir = self.forecast_output_dir / 'diagnostics'

        self.log_dir = BASE_DIR / 'logs'

    def make_dirs(self):
        for directory in [
            self.regression_output_dir, self.forecast_output_dir,
            self.regression_coefficient_dir, self.regression_diagnostic_dir,
            self.forecast_draw_dir, self.forecast_diagnostic_dir,
            self.log_dir
        ]:
            if directory is not None:
                os.makedirs(str(directory), exist_ok=True)

    def location_draw_forecast_file(self, location_id, draw_id):
        os.makedirs(self.forecast_output_dir / str(location_id), exist_ok=True)
        return self.forecast_output_dir / str(location_id) / f'draw_{draw_id}.csv'

    def get_infection_file(self, location_id, draw_id):
        folder = _get_infection_folder_from_location_id(location_id, self.infection_dir)
        return self.infection_dir / folder / INFECTION_FILE_PATTERN.format(draw_id=draw_id)

    def get_covariate_file(self):
        return self.covariate_dir / COVARIATE_FILE

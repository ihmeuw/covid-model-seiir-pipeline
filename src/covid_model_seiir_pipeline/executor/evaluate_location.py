import logging
from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import pandas as pd

from covid_model_seiir_pipeline.core.versioner import Directories, _get_infection_folder_from_location_id
from covid_model_seiir_pipeline.core.versioner import load_ode_settings
from covid_model_seiir_pipeline.core.versioner import INFECTION_FILE_PATTERN


log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    log.info("parsing arguments")

    parser = ArgumentParser()
    parser.add_argument("--version-name", type=str, required=True)
    parser.add_argument("--location-id", type=int, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    return args


def load_validation_data(directories, location_id):
    folder = _get_infection_folder_from_location_id(location_id, directories.infection_dir)
    file = directories.infection_dir / folder / f'VALIDATION_{INFECTION_FILE_PATTERN.format(draw_id=0)}'
    df = pd.read_csv(file)
    return df


def read_data(version_name, location_id):

    directories = Directories(
        ode_version=version_name,
        regression_version=version_name,
        forecast_version=version_name
    )
    forecast = pd.read_csv(
        directories.location_output_forecast_file(
            location_id=location_id, forecast_type='deaths'
        )
    )
    columns = forecast.columns
    draw_cols = [x for x in columns if 'draw' in x]

    data = load_validation_data(
        directories=directories,
        location_id=location_id
    )


def main():

    args = parse_arguments()
    read_data(
        version_name=args.version_name,
        location_id=args.location_id
    )


if __name__ == '__main__':
    main()

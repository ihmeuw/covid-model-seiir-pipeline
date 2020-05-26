import logging
from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import pandas as pd
import numpy as np
from pathlib import Path

from covid_model_seiir_pipeline.core.versioner import Directories, _get_infection_folder_from_location_id
from covid_model_seiir_pipeline.core.versioner import INFECTION_FILE_PATTERN
from covid_model_seiir_pipeline.core.versioner import INFECTION_COL_DICT


log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    log.info("parsing arguments")

    parser = ArgumentParser()
    parser.add_argument("--version-name", type=str, required=True)
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

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


def calculate_metrics(df, observed_col, draw_cols):
    observed = df[observed_col].values
    draws = np.asarray(df[draw_cols])
    mean = draws.mean(axis=1)
    ui = np.quantile(draws, axis=1, q=[0.025, 0.975])

    df['observed'] = observed
    df['mean'] = mean
    df['lower'] = ui[0, :]
    df['upper'] = ui[1, :]
    df['bias'] = mean - observed
    df['sq_error'] = df['bias'] ** 2
    df['coverage'] = np.logical_and(ui[0, :] <= observed, observed <= ui[1, :])

    return df[[
        INFECTION_COL_DICT['COL_LOC_ID'], INFECTION_COL_DICT['COL_DATE'],
        'observed', 'mean', 'lower', 'upper', 'bias', 'sq_error', 'coverage'
    ]]


def read_data(version_name, location_id, output_dir):

    output_dir = Path(output_dir)

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
    forecast = forecast[[INFECTION_COL_DICT['COL_DATE']] + draw_cols]

    data = load_validation_data(
        directories=directories,
        location_id=location_id
    )
    df = data.merge(forecast, on=INFECTION_COL_DICT['COL_DATE'])
    metrics = calculate_metrics(df, observed_col=INFECTION_COL_DICT['COL_DEATHS_DATA'], draw_cols=draw_cols)
    metrics.to_csv(output_dir / f'metrics_{location_id}.csv', index=False)


def main():

    args = parse_arguments()
    read_data(
        version_name=args.version_name,
        location_id=args.location_id,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

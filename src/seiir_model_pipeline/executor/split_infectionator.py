import numpy as np
import pandas as pd
import shlex
import os
from argparse import ArgumentParser, Namespace
from typing import Optional
import logging
from pathlib import Path

from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT

log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--param-dir", type=str, required=True)
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--time-holdout", type=int, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def get_train_test_data(df, time_holdout, day_shift):

    col_date = INFECTION_COL_DICT['COL_DATE']
    col_obs = INFECTION_COL_DICT['COL_OBS_DEATHS']
    
    df = df.loc[df[col_obs] == 1].copy()

    holdout_days = np.timedelta64(time_holdout, 'D')
    end_date = np.datetime64(df[col_date].max(), 'D')
    split_date = end_date - holdout_days

    df[col_date] = pd.to_datetime(df[col_date])

    train_dates = pd.to_datetime(df[col_date]) <= split_date
    tests_dates = pd.to_datetime(df[col_date]) > split_date

    train = df.loc[train_dates]
    tests = df.loc[tests_dates]

    train_observed = pd.to_datetime(train[col_date]) <= split_date - day_shift

    train[col_obs] = 1
    train.loc[train_observed, col_obs] = 0
    tests[col_obs] = 0

    return train, tests


class NoParametersError(Exception):
    pass


def get_day_shift(param_directory, draw):
    try:
        params = pd.read_csv(param_directory / f'params_draw_{draw}.csv')
        return params.loc[params.params == 'day_shift', 'values'].iloc[0]
    except FileNotFoundError:
        raise NoParametersError


def create_new_files(old_directory, new_directory, param_directory, location_id, time_holdout):
    old_directory = Path(old_directory)
    new_directory = Path(new_directory)

    location_folders = os.listdir(old_directory)
    folder = [
        x for x in location_folders
        if os.path.isdir(old_directory / x) and
        int(x.split("_")[-1]) == location_id
    ]
    assert len(folder) == 1, f"More than one infectionator folder for location ID {location_id}!"
    folder = folder[0]

    if os.path.exists(new_directory / folder):
        raise ValueError(f"Copying for location {location_id} has already been done!")
    else:
        os.makedirs(new_directory / folder, exist_ok=False)
        files = os.listdir(old_directory / folder)
        for f in files:
            draw = int(f.split('_')[0].split('draw')[-1])
            try:
                day_shift = get_day_shift(param_directory, draw)
            except NoParametersError:
                continue
            df = pd.read_csv(old_directory / folder / f)
            train, test = get_train_test_data(df, time_holdout, day_shift)
            if os.path.exists(new_directory / folder / f):
                log.warning(f"{str(new_directory / folder / f)} already exists.")
            else:
                train.to_csv(new_directory / folder / f, index=False)
                test.to_csv(new_directory / folder / f'VALIDATION_{f}', index=False)


def main():

    args = parse_arguments()
    create_new_files(
        old_directory=args.input_dir,
        new_directory=args.output_dir,
        param_directory=args.param_dir,
        location_id=args.location_id,
        time_holdout=args.time_holdout
    )


if __name__ == '__main__':
    main()

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
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--time-holdout", type=int, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def get_train_test_data(df, time_holdout):

    col_date = INFECTION_COL_DICT['COL_DATE']
    col_obs = INFECTION_COL_DICT['COL_OBS_DEATHS']

    holdout_days = np.timedelta64(time_holdout, 'D')
    end_date = np.datetime64(df.loc[df[col_obs] == 1, col_date].max(), 'D')
    split_date = end_date - holdout_days

    df[col_date] = pd.to_datetime(df[col_date])

    train_dates = pd.to_datetime(df[col_date]) <= split_date
    tests_dates = pd.to_datetime(df[col_date]) > split_date

    train = df.loc[train_dates]
    tests = df.loc[tests_dates]

    return train, tests


def create_new_files(old_directory, new_directory, location_id, time_holdout):

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
            df = pd.read_csv(old_directory / folder / f)
            train, test = get_train_test_data(df, time_holdout)
            if os.path.exists(new_directory / folder / f):
                log.warning(f"{str(new_directory / folder / f)} already exists.")
            else:
                train.to_csv(new_directory / folder / f, index=False)
                test.to_csv(new_directory / folder / f'VALIDATION_{f}', index=False)


def main():

    args = parse_arguments()
    create_new_files(
        old_directory=args.old_directory,
        new_directory=args.new_directory,
        location_id=args.location_id,
        time_holdout=args.time_holdout
    )


if __name__ == '__main__':
    main()

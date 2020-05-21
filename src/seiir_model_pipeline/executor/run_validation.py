import numpy as np
import pandas as pd
import shlex
import os
from argparse import ArgumentParser, Namespace
from typing import Optional
import logging

from seiir_model_pipeline.core.utils import clone_run
from seiir_model_pipeline.core.versioner import BASE_DIR, Directories, OUTPUT_DIR, INFECTION_COL_DICT, load_ode_settings
from seiir_model_pipeline.executor.run import run

log = logging.getLogger(__name__)

VALIDATION_INPUT_DIR = BASE_DIR / 'seir-validations'


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--version-name", type=int, required=True)
    parser.add_argument("--time-holdout", type=int, required=True)
    parser.add_argument("--validation-output-dir", type=str, required=False)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def get_train_test_data(df, time_holdout):

    col_date = INFECTION_COL_DICT['col_date']

    holdout_days = np.timedelta64(time_holdout, 'D')
    end_date = np.datetime64(df[col_date].max(), 'D')
    split_date = end_date - holdout_days

    df[col_date] = pd.to_datetime(df[col_date])

    train_dates = pd.to_datetime(df[col_date]) <= split_date
    tests_dates = pd.to_datetime(df[col_date]) > split_date

    train = df.loc[train_dates]
    tests = df.loc[tests_dates]

    return train, tests


def get_validation_version_name(original_version, time_holdout):
    new_version = f'{original_version}.validation.HO{time_holdout}'
    dirs = os.listdir(OUTPUT_DIR / 'ode')
    match = [x for x in dirs if f'validation.HO{time_holdout}' in x]
    version = len(match) + 1
    new_version += f'.{version:02}'
    return new_version


def create_validation_infection_version(original_version, infection_version, time_holdout):

    directories = Directories(ode_version=original_version)
    new_version = f'{infection_version}.validate.{time_holdout}'

    old_directory = directories.infection_dir
    new_directory = VALIDATION_INPUT_DIR / new_version

    if os.path.exists(new_directory):
        log.info(f"Validation version {new_version} already exists.")
    else:
        dirs = os.listdir(old_directory)
        for d in dirs:
            os.makedirs(new_directory / d)
            files = os.listdir(old_directory / d)
            for f in files:
                df = pd.read_csv(old_directory / d / f)
                train, test = get_train_test_data(df, time_holdout)
                if os.path.exists(new_directory / d / f):
                    log.warning(f"{str(new_directory / d / f)} already exists.")
                else:
                    train.to_csv(new_directory / d / f, index=False)
                    test.to_csv(new_directory / d / f'VALIDATION_{f}')

    return new_version


def launch_validation(version_name, time_holdout):
    log.info(f"Cloning {version_name} for a validation run with {time_holdout} holdout days.")

    new_version_name = get_validation_version_name(original_version=version_name, time_holdout=time_holdout)
    infection_version = load_ode_settings(version_name).infection_version

    new_infection_version = create_validation_infection_version(
        original_version=version_name,
        infection_version=infection_version,
        time_holdout=time_holdout
    )
    clone_run(
        old_version=version_name,
        new_version=new_version_name,
        infection_dir=VALIDATION_INPUT_DIR,
        infection_version=new_infection_version
    )
    run(
        ode_version=new_version_name,
        regression_version=new_version_name,
        forecast_version=new_version_name,
        run_splicer=True,
        create_diagnostics=True
    )


def run_validation_analysis(version_name, output_path):
    pass


def main():

    args = parse_arguments()
    launch_validation(
        version_name=args.version_name,
        time_holdout=args.time_holdout
    )
    if isinstance(args.validation_output_dir, str):
        run_validation_analysis(
            version_name=args.version_name,
            output_path=args.validation_output_dir
        )


if __name__ == '__main__':
    main()


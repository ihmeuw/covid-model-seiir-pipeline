import logging
from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import pandas as pd
import numpy as np
import os
from pathlib import Path

from covid_model_seiir_pipeline.core.versioner import INFECTION_COL_DICT

log = logging.getLogger(__name__)

ALL_METRICS_FILE = 'ALL_metrics.csv'
COL_DAYS = 'days'


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    log.info("parsing arguments")

    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    return args


def date_to_days(date):
    date = pd.to_datetime(date)
    days = date - date.min() + np.timedelta64(1, 'D')
    return days


def load_all_validation_data(output_dir):
    dfs = []
    for file in os.listdir(output_dir):
        if file == ALL_METRICS_FILE:
            continue
        df = pd.read_csv(output_dir / file)
        df[COL_DAYS] = date_to_days(df[INFECTION_COL_DICT['COL_DATE']])
        df.drop(columns=INFECTION_COL_DICT['COL_DATE'], inplace=True, axis=1)
        dfs.append(df)
    return pd.concat(dfs).reset_index()


def collapse_metrics(df):
    groups = df.groupby('days')

    rmse = groups.sq_error.apply(lambda x: np.sqrt(np.mean(x))).values
    bias = groups.bias.mean().values
    coverage = groups.coverage.mean().values

    result = pd.DataFrame({
        COL_DAYS: list(groups.groups.keys()),
        'rmse': rmse,
        'bias': bias,
        'coverage': coverage
    })

    return result


def read_data(output_dir):
    output_dir = Path(output_dir)

    df = load_all_validation_data(output_dir)
    metrics = collapse_metrics(df)
    metrics.to_csv(output_dir / ALL_METRICS_FILE, index=False)


def main():
    args = parse_arguments()
    read_data(
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

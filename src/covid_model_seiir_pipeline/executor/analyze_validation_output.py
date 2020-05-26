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


def load_all_validation_data(output_dir):
    dfs = []
    for file in os.listdir(output_dir):
        if file == ALL_METRICS_FILE:
            continue
        df = pd.read_csv(output_dir / file)
        dfs.append(df)
    return pd.concat(dfs).reset_index()


def collapse_metrics(df):
    groups = df.groupby(INFECTION_COL_DICT['COL_DATE'])

    rmse = groups.sq_error.apply(lambda x: np.sqrt(np.mean(x))).values
    bias = groups.bias.mean().values
    coverage = groups.coverage.mean().values

    result = pd.DataFrame({
        INFECTION_COL_DICT['COL_DATE']: list(groups.groups.keys()),
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

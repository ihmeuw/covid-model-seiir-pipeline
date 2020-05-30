from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Optional

from loguru import logger
import numpy as np
import pandas as pd
import tqdm


def compare_fit_beta(old_root: str, new_root: str, output_dir: str):
    logger.info(f'Comparing {old_root} to {new_root}')
    old_root = Path(old_root)
    new_root = Path(new_root)
    output_dir = Path(output_dir)

    old_locs = [p.name for p in old_root.iterdir()]
    new_locs = [p.name for p in new_root.iterdir()]

    missing = set(old_locs).difference(new_locs)
    extra = set(new_locs).difference(old_locs)

    if not (missing or extra):
        logger.info('all locations found in both root directories.')
    else:
        if missing:
            logger.info(f'new root is missing locations {missing}')
        if extra:
            logger.info(f'new root has extra locations {extra}')

    locs = set(old_locs).intersection(new_locs)

    old_data = []
    new_data = []

    logger.info('Gathering files')
    old_files = [loc_draw_file for loc in locs for loc_draw_file in (old_root / loc).iterdir()]
    new_files = [loc_draw_file for loc in locs for loc_draw_file in (new_root / loc).iterdir()]

    logger.info('Loading old data.')
    for loc_draw_file in tqdm.tqdm(old_files):
        df = pd.read_csv(loc_draw_file)
        df['draw'] = int(loc_draw_file.stem.split('_')[-1])
        old_data.append(df)
    logger.info('concatting old data.')
    old_data = pd.concat(old_data)
    old_data = old_data.set_index(['loc_id', 'draw', 'days']).sort_index()

    logger.info('Loading new data.')
    for loc_draw_file in tqdm.tqdm(new_files):
        df = pd.read_csv(loc_draw_file)
        df['draw'] = int(loc_draw_file.stem.split('_')[-1])
        new_data.append(df)
    logger.info('concatting new data.')
    new_data = pd.concat(new_data)
    new_data = new_data.set_index(['loc_id', 'draw', 'days']).sort_index()

    for col in old_data.columns:
        old_col = old_data[col]
        new_col = new_data[col]
        exact_equal = (old_col == new_col).sum()
        almost_equal = np.sum(np.isclose(old_col, new_col, atol=1e-6, rtol=.1, equal_nan=True))

        logger.info(f'Column {col} summary:')
        logger.info(f'size: {len(old_col)}')
        logger.info(f'exact: {exact_equal}')
        logger.info(f'almost: {almost_equal}')

    old_data.to_csv(output_dir / 'old.csv', index=False)
    new_data.to_csv(output_dir / 'new.csv', index=False)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--old-root", type=str, required=True)
    parser.add_argument("--new-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    compare_fit_beta(args.old_root, args.new_root, args.output_dir)


if __name__ == '__main__':
    main()

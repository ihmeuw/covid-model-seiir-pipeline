import logging
from argparse import ArgumentParser

import pandas as pd

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from seiir_model_pipeline.core.data import load_all_location_data
from seiir_model_pipeline.core.splicer import Splicer

log = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--regression-version", typt=int, required=True)
    parser.add_argument("--forecast-version", type=int, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR splicing.")

    # Load metadata
    directories = args_to_directories(args)
    regression_settings = load_regression_settings(args.regression_settings)
    forecast_settings = load_forecast_settings(args.forecast_version)

    spliced_data = pd.DataFrame()
    splicer = Splicer()

    for draw_id in range(regression_settings.n_draws):
        print(f"On draw {draw_id}.")
        location_data = load_all_location_data(
            directories, location_ids=[args.location_id], draw_id=draw_id
        )

    spliced_data.to_csv(
        directories.location_output_forecast_file(location_id=args.location_id)
    )

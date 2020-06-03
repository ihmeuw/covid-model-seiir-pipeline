import logging
from argparse import ArgumentParser

import pandas as pd

from covid_model_seiir_pipeline.core.versioner import args_to_directories
from covid_model_seiir_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from covid_model_seiir_pipeline.core.data import load_all_location_data, load_component_forecasts
from covid_model_seiir_pipeline.core.data import load_beta_fit, load_beta_params
from covid_model_seiir_pipeline.core.splicer import Splicer

log = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR splicing.")

    # Load metadata
    directories = args_to_directories(args)
    regression_settings = load_regression_settings(args.regression_version)
    forecast_settings = load_forecast_settings(args.forecast_version)

    splicer = Splicer(n_draws=regression_settings.n_draws, location_id=args.location_id)
    splicer.capture_location_name(
        metadata_path=directories.get_location_metadata_file(regression_settings.location_set_version_id)
    )

    for draw_id in range(regression_settings.n_draws):
        print(f"On draw {draw_id}.")
        infection_data = load_all_location_data(
            directories, location_ids=[args.location_id], draw_id=draw_id
        )
        component_fit = load_beta_fit(
            directories, draw_id=draw_id,
            location_id=args.location_id
        )
        params = load_beta_params(directories, draw_id=draw_id)
        component_forecasts = load_component_forecasts(
            directories, location_id=args.location_id, draw_id=draw_id
        )
        splicer.splice_draw(
            infection_data[args.location_id],
            component_fit,
            component_forecasts,
            params,
            draw_id=draw_id
        )

    splicer.save_cases(directories.location_output_forecast_file(location_id=args.location_id, forecast_type='cases'))
    splicer.save_deaths(directories.location_output_forecast_file(location_id=args.location_id, forecast_type='deaths'))
    splicer.save_reff(directories.location_output_forecast_file(location_id=args.location_id, forecast_type='reff'))


if __name__ == '__main__':
    main()

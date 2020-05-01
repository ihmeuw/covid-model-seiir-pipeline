from argparse import ArgumentParser
import logging
from typing import List

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from seiir_model_pipeline.core.data import load_covariates
from seiir_model_pipeline.core.data import load_mr_coefficients

log = logging.getLogger(__name__)


def get_args():
    """
    Gets arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR beta forecasting.")
    log.info("Running for location {args.location_id}, scenario {args.scenario_id}.")

    # Load metadata
    directories = args_to_directories(args)
    regression_settings = load_regression_settings(directories)
    forecast_settings = load_forecast_settings(directories)

    coefficients = load_mr_coefficients(directories)
    covariate_forecasts = load_covariates(
        directories, covariate_names=list(regression_settings.covariates.keys())
    )

    mr = ModelRunner()

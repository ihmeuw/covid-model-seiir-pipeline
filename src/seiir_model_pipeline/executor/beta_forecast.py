from argparse import ArgumentParser
import logging
from typing import List

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from seiir_model_pipeline.core.versioner import PEAK_DATE_FILE, PEAK_DATE_COL_DICT, COVARIATE_COL_DICT, OBSERVED_DICT
from seiir_model_pipeline.core.data import load_covariates
from seiir_model_pipeline.core.data import load_mr_coefficients
from seiir_model_pipeline.core.utils import convert_to_covmodel
from seiir_model_pipeline.core.versioner import OBSERVED_DICT

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
    regression_settings = load_regression_settings(args.regression_version)
    forecast_settings = load_forecast_settings(args.forecast_version)
    covmodel_set = convert_to_covmodel(regression_settings.covariates)
    covariate_data = load_covariates(
        directories,
        location_id=[args.location_id],
        col_loc_id=COVARIATE_COL_DICT['COL_LOC_ID'],
        col_observed=COVARIATE_COL_DICT['COL_OBSERVED'],
        forecasted=OBSERVED_DICT['forecasted']
    )
    mr = ModelRunner()
    regression_fit = load_mr_coefficients(
        directories=directories,
        location_id=args.location_id,
        draw_id=args.draw_id
    )
    forecasts = mr.predict_beta_forward(
        covmodel_set=covmodel_set,
        df_cov=covariate_data,
        df_cov_coef=regression_fit,
        col_t=COVARIATE_COL_DICT['COL_DATE'],
        col_group=COVARIATE_COL_DICT['COL_LOC_ID']
    )


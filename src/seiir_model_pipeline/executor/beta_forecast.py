from argparse import ArgumentParser
import logging
from typing import List

from seiir_model.model_runner import ModelRunner
from seiir_model.ode_forecasting.ode_runner import SiierdModelSpecs

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from seiir_model_pipeline.core.versioner import PEAK_DATE_FILE, PEAK_DATE_COL_DICT, COVARIATE_COL_DICT, OBSERVED_DICT
from seiir_model_pipeline.core.data import load_covariates, load_beta_fit, load_beta_params
from seiir_model_pipeline.core.data import load_mr_coefficients
from seiir_model_pipeline.core.utils import convert_to_covmodel
from seiir_model_pipeline.core.versioner import OBSERVED_DICT
from seiir_model_pipeline.core.utils import get_ode_init_cond

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

    mr = ModelRunner()

    # Get all inputs for the beta forecasting
    covmodel_set = convert_to_covmodel(regression_settings.covariates)
    covariate_data = load_covariates(
        directories,
        location_id=args.location_id,
        col_loc_id=COVARIATE_COL_DICT['COL_LOC_ID'],
        col_observed=COVARIATE_COL_DICT['COL_OBSERVED'],
        forecasted=OBSERVED_DICT['forecasted']
    )
    regression_fit = load_mr_coefficients(
        directories=directories,
        location_id=args.location_id,
        draw_id=args.draw_id
    )
    forecasts = mr.predict_beta_forward(
        covmodel_set=covmodel_set,
        df_cov=covariate_data,
        col_t=COVARIATE_COL_DICT['COL_DATE'],
        col_group=COVARIATE_COL_DICT['COL_LOC_ID']
    )

    # Get all inputs for the ODE
    beta_fit = load_beta_fit(
        directories, draw_id=args.draw_id,
        location_id=args.location_id
    )
    beta_params = load_beta_params(
        directories, draw_id=args.draw_id
    )
    init_cond = get_ode_init_cond(
        beta_ode_fit=beta_fit,
        current_date=covariate_data[COVARIATE_COL_DICT['COL_DATE']].min()
    )
    model_specs = SiierdModelSpecs(
        alpha=beta_params['alpha'],
        sigma=beta_params['beta']
    )
    times = None
    mr.forecast(
        model_specs=model_specs,
        init_cond=None,
        times=times,
        betas=forecasts,
        dt=regression_settings.solver_dt
    )


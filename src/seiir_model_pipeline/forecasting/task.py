from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import logging

import pandas as pd
import numpy as np

from seiir_model.model_runner import ModelRunner
from seiir_model.ode_forecasting.ode_runner import SiierdModelSpecs

from seiir_model_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT, COVARIATE_COL_DICT
from seiir_model_pipeline.core.versioner import Directories

from seiir_model_pipeline.core.data import load_covariates, load_beta_fit, load_beta_params
from seiir_model_pipeline.core.data import load_mr_coefficients

from seiir_model_pipeline.core.model_inputs import convert_to_covmodel
from seiir_model_pipeline.core.model_inputs import get_ode_init_cond

from seiir_model_pipeline.core.utils import date_to_days

log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--location-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--coefficient-version", type=str, required=False, default=None)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def run_beta_forecast(location_id: int, regression_version: str, forecast_version: str,
                      coefficient_version: str = None):

    log.info("Initiating SEIIR beta forecasting.")

    # -------------------------- LOAD INPUTS -------------------- #
    # Load metadata
    directories = Directories(
        regression_version=regression_version,
        forecast_version=forecast_version
    )
    regression_settings = load_regression_settings(regression_version)
    forecast_settings = load_forecast_settings(forecast_version)

    # -------------------------- FORECAST THE BETA FORWARDS -------------------- #
    mr = ModelRunner()

    # Get all inputs for the beta forecasting
    # Get all inputs for the ODE
    scales = []

    for draw_id in range(regression_settings.n_draws):
        print(f"On draw {draw_id}\n")

        # Load the previous beta fit compartments and ODE parameters
        beta_fit = load_beta_fit(
            directories, draw_id=draw_id,
            location_id=location_id
        )
        beta_params = load_beta_params(
            directories, draw_id=draw_id
        )

        # Convert settings to the covariates model and load covariates data
        _, all_covmodels_set = convert_to_covmodel(
            regression_settings.covariates,
            regression_settings.covariates_order,
        )
        covariate_data = load_covariates(
            directories,
            covariate_version=forecast_settings.covariate_version,
            location_ids=[location_id]
        )

        # Figure out what date we need to forecast from (the end of the component fit in regression task)
        beta_fit_date = pd.to_datetime(beta_fit[INFECTION_COL_DICT['COL_DATE']])
        CURRENT_DATE = beta_fit[beta_fit_date == beta_fit_date.max()][INFECTION_COL_DICT['COL_DATE']].iloc[0]
        covariate_date = pd.to_datetime(covariate_data[COVARIATE_COL_DICT['COL_DATE']])
        covariate_data = covariate_data.loc[covariate_date >= beta_fit_date.max()].copy()

        # Load the regression coefficients
        regression_fit = load_mr_coefficients(
            directories=directories,
            draw_id=draw_id
        )
        # Forecast the beta forward with those coefficients
        forecasts = mr.predict_beta_forward_prod(
            covmodel_set=all_covmodels_set,
            df_cov=covariate_data,
            df_cov_coef=regression_fit,
            col_t=COVARIATE_COL_DICT['COL_DATE'],
            col_group=COVARIATE_COL_DICT['COL_LOC_ID']
        )

        betas = forecasts.beta_pred.values
        days = forecasts[COVARIATE_COL_DICT['COL_DATE']].values
        times = date_to_days(days)

        # Anchor the betas at the last observed beta (fitted)
        # and scale everything into the future from this anchor value
        anchor_beta = beta_fit.beta[beta_fit.date == CURRENT_DATE].iloc[0]
        scale = anchor_beta / betas[0]
        scales.append(scale)
        # scale = scale + (1 - scale)/20.0*np.arange(betas.size)
        # scale[21:] = 1.0
        betas = betas * scale

        # Get initial conditions based on the beta fit for forecasting into the future
        init_cond = get_ode_init_cond(
            beta_ode_fit=beta_fit,
            current_date=CURRENT_DATE,
            location_id=location_id
        ).astype(float)
        N = np.sum(init_cond)  # total population
        model_specs = SiierdModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=N
        )
        # Forecast all of the components based on the forecasted beta
        forecasted_components = mr.forecast(
            model_specs=model_specs,
            init_cond=init_cond,
            times=times,
            betas=betas,
            dt=regression_settings.solver_dt
        )
        forecasted_components[COVARIATE_COL_DICT['COL_DATE']] = days
        forecasted_components.to_csv(
            directories.location_draw_component_forecast_file(
                location_id=location_id,
                draw_id=draw_id
            )
        )
    df_scales = pd.DataFrame({
        'beta_scales': scales
    })
    df_scales.to_csv(
        directories.location_beta_scaling_file(
            location_id=location_id
        ),
        index=False
    )


def main():
    args = parse_arguments()
    run_beta_forecast(location_id=args.location_id,
                      regression_version=args.regression_version,
                      forecast_version=args.forecast_version,
                      coefficient_version=args.coefficient_version)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import logging

import pandas as pd
import numpy as np

from covid_model_seiir.model_runner import ModelRunner
from covid_model_seiir.ode_forecasting.ode_runner import SiierdModelSpecs

from covid_model_seiir_pipeline.core.versioner import load_forecast_settings, load_regression_settings
from covid_model_seiir_pipeline.core.versioner import INFECTION_COL_DICT, COVARIATE_COL_DICT
from covid_model_seiir_pipeline.core.versioner import Directories

from covid_model_seiir_pipeline.core.data import load_covariates, load_beta_fit, load_beta_params
from covid_model_seiir_pipeline.core.data import load_mr_coefficients

from covid_model_seiir_pipeline.core.model_inputs import convert_to_covmodel
from covid_model_seiir_pipeline.core.model_inputs import get_ode_init_cond

from covid_model_seiir_pipeline.core.utils import date_to_days, beta_shift

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
    # You now have access to forecast_settings.functionally_immune_proportion or whatever you call it.
    if forecast_settings.theta_locations_file:
        theta_locations = pd.read_csv(forecast_settings.theta_locations_file)
        theta_locations.to_csv(
            directories.forecast_output_dir / 'theta.csv', index=False
        )
        theta_locations = theta_locations.set_index("location_id").theta
        theta = theta_locations.get(location_id, default=0)
    else:
        theta = forecast_settings.theta

    beta_shift_dict = regression_settings.beta_shift_dict
    if 'residual_offset_file' in beta_shift_dict:
        residual_offset = pd.read_csv(beta_shift_dict['residual_offset_file']).set_index('location_id')
        beta_shift_dict['total_deaths'] = residual_offset['total_deaths'].get(location_id, default=None)
        beta_shift_dict['offset'] = residual_offset['residual_mean'].get(location_id, default=0)
        del beta_shift_dict['residual_offset_file']
    else:
        beta_shift_dict['total_deaths'] = None
        beta_shift_dict['offset'] = 0

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
            location_ids=[location_id],
            draw_id=draw_id if any(forecast_settings.covariate_draw_dict.values()) else None
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

        betas, scale_params = beta_shift(beta_fit, betas, draw_id, **beta_shift_dict)
        scale_params['draw'] = draw_id
        scales.append(scale_params)

        # Get initial conditions based on the beta fit for forecasting into the future
        init_cond = get_ode_init_cond(
            beta_ode_fit=beta_fit,
            current_date=CURRENT_DATE,
            location_id=location_id
        ).astype(float)

        # Kirsten to do math here

        N = np.sum(init_cond)  # total population
        model_specs = SiierdModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=N
        )
        # make theta the same length as betas
        thetas = np.repeat(theta, betas.size)
        # Forecast all of the components based on the forecasted beta
        forecasted_components = mr.forecast(
            model_specs=model_specs,
            init_cond=init_cond,
            times=times,
            betas=betas,
            thetas=thetas,
            dt=regression_settings.solver_dt
        )
        forecasted_components[COVARIATE_COL_DICT['COL_DATE']] = days
        forecasted_components.to_csv(
            directories.location_draw_component_forecast_file(
                location_id=location_id,
                draw_id=draw_id
            )
        )

    scales_flat = {}
    for scale_param in scales[0]:
        scales_flat[scale_param] = []
        for scale_param_dict in scales:
            scales_flat[scale_param].append(scale_param_dict[scale_param])

    df_scales = pd.DataFrame(scales_flat)
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

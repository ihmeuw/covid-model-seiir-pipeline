from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import logging
import numpy as np
import pandas as pd

from covid_model_seiir.model_runner import ModelRunner

from covid_model_seiir_pipeline.core.versioner import COVARIATE_COL_DICT, INFECTION_COL_DICT
from covid_model_seiir_pipeline.core.versioner import load_regression_settings
from covid_model_seiir_pipeline.core.versioner import Directories

from covid_model_seiir_pipeline.core.data import load_all_location_data
from covid_model_seiir_pipeline.core.data import load_mr_coefficients
from covid_model_seiir_pipeline.core.data import load_covariates

from covid_model_seiir_pipeline.core.utils import load_locations

from covid_model_seiir_pipeline.core.model_inputs import process_ode_process_input
from covid_model_seiir_pipeline.core.model_inputs import convert_inputs_for_beta_model
from covid_model_seiir_pipeline.core.model_inputs import convert_to_covmodel

log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def run_beta_regression(draw_id: int, regression_version: str):

    # -------------------------- LOAD INPUTS -------------------- #
    # Load metadata
    directories = Directories(
        regression_version=regression_version,
        forecast_version=None
    )
    settings = load_regression_settings(regression_version)

    # Save thetas file, if one exists
    if settings.theta_locations_file:
        theta_locations = pd.read_csv(settings.theta_locations_file)
        theta_locations.to_csv(
            directories.regression_output_dir / 'theta.csv', index=False
        )

    # Load data
    location_ids = load_locations(directories)
    location_data = load_all_location_data(
        directories=directories, location_ids=location_ids, draw_id=draw_id
    )
    covariate_data = load_covariates(
        directories,
        covariate_version=settings.covariate_version,
        location_ids=location_ids,
        draw_id=draw_id if any(settings.covariate_draw_dict.values()) else None
    )

    # This seed is so that the alpha, sigma, gamma1 and gamma2 parameters are reproducible
    np.random.seed(draw_id)
    beta_fit_inputs = process_ode_process_input(
        settings=settings,
        location_data=location_data,
    )

    # ----------------------- BETA ODE -------------------------------- #
    # Start a Model Runner with the processed inputs and fit the beta  ODE
    mr = ModelRunner()
    mr.fit_beta_ode(beta_fit_inputs)

    # -------------- BETA REGRESSION WITH LOADED COVARIATES -------------------- #
    # Convert inputs for beta regression using model_inputs utilities functions
    ordered_covmodel_sets, all_covmodels_set = convert_to_covmodel(
        cov_dict=settings.covariates,
        cov_order_list=settings.covariates_order
    )
    mr_data = convert_inputs_for_beta_model(
        data_cov=(
            covariate_data, COVARIATE_COL_DICT['COL_DATE'],
            COVARIATE_COL_DICT['COL_LOC_ID']
        ),
        df_beta=mr.get_beta_ode_fit(),
        covmodel_set=all_covmodels_set,
    )
    if settings.coefficient_version is not None:
        # If you want to use a specific coefficient version,
        # this will read in the coefficients and then they will be
        # passed to the beta regression.

        coefficient_directory = Directories(regression_version=settings.coefficient_version)
        fixed_coefficients = load_mr_coefficients(
            directories=coefficient_directory,
            draw_id=draw_id
        )
    else:
        fixed_coefficients = None
    # Fit the beta regression; the `path` argument automatically saves the coefficients
    # to the specified file
    mr.fit_beta_regression_prod(
        ordered_covmodel_sets=ordered_covmodel_sets,
        mr_data=mr_data,
        path=directories.get_draw_coefficient_file(draw_id),
        df_cov_coef=fixed_coefficients,
        sequential=settings.sequential
    )
    # -------------------- POST PROCESSING AND SAVING ------------------------ #
    # Get the fitted values of beta from the regression model and append on
    # to the fits -- **this is just for diagnostic purposes**
    regression_fit = load_mr_coefficients(
        directories=directories,
        draw_id=draw_id
    )
    forecasts = mr.predict_beta_forward_prod(
        covmodel_set=all_covmodels_set,
        df_cov=covariate_data,
        df_cov_coef=regression_fit,
        col_t=COVARIATE_COL_DICT['COL_DATE'],
        col_group=COVARIATE_COL_DICT['COL_LOC_ID']
    )
    beta_fit = mr.get_beta_ode_fit()
    regression_betas = forecasts[
        [COVARIATE_COL_DICT['COL_LOC_ID'], COVARIATE_COL_DICT['COL_DATE']] +
        list(settings.covariates.keys()) + ['beta_pred']
    ]
    beta_fit_covariates = beta_fit.merge(
        regression_betas,
        left_on=[INFECTION_COL_DICT['COL_LOC_ID'], INFECTION_COL_DICT['COL_DATE']],
        right_on=[COVARIATE_COL_DICT['COL_LOC_ID'], COVARIATE_COL_DICT['COL_DATE']],
        how='left'
    )
    # Save location-specific beta fit (compartment) files for easy reading later
    for l_id in location_ids:
        loc_beta_fits = beta_fit_covariates.loc[beta_fit_covariates[INFECTION_COL_DICT['COL_LOC_ID']] == l_id].copy()
        loc_beta_fits.to_csv(directories.get_draw_beta_fit_file(l_id, draw_id), index=False)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    mr.get_beta_ode_params().to_csv(directories.get_draw_beta_param_file(draw_id), index=False)


def main():

    args = parse_arguments()
    run_beta_regression(draw_id=args.draw_id, regression_version=args.regression_version)


if __name__ == '__main__':
    main()

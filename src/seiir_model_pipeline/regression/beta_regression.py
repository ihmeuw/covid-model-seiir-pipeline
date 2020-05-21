from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional

import numpy as np

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.regression.specification import load_regression_specification
from seiir_model_pipeline.regression.data import RegressionData, InfectionData
from seiir_model_pipeline.regression.globals import COVARIATE_COL_DICT, INFECTION_COL_DICT

from seiir_model_pipeline.core.model_inputs import process_ode_process_input
from seiir_model_pipeline.core.model_inputs import convert_inputs_for_beta_model
from seiir_model_pipeline.core.model_inputs import convert_to_covmodel_regression

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
    regression_specification = load_regression_specification(regression_version)
    regression_data = RegressionData(
        regression_dir=Path(regression_specification.data.output_root)
    )
    infection_data = InfectionData(
        infection_dir=Path(regression_specification.data.infection_version),
    )

    # Load data
    location_ids = regression_data.location_ids

    location_data = infection_data.load_all_location_data(draw_id=draw_id)
    covariate_data = regression_data.load_covariates(draw_id=None)

    # This seed is so that the alpha, sigma, gamma1 and gamma2 parameters are reproducible
    np.random.seed(draw_id)
    beta_fit_inputs = process_ode_process_input(
        settings=regression_specification.parameters,
        location_data=location_data
    )

    # ----------------------- BETA SPLINE + ODE -------------------------------- #
    # Start a Model Runner with the processed inputs and fit the beta spline / ODE
    mr = ModelRunner()
    mr.fit_beta_ode(beta_fit_inputs)

    # -------------- BETA REGRESSION WITH LOADED COVARIATES -------------------- #
    # Convert inputs for beta regression using model_inputs utilities functions
    ordered_covmodel_sets, all_covmodels_set = convert_to_covmodel_regression(
        covariates=regression_specification.covariates
    )
    mr_data = convert_inputs_for_beta_model(
        data_cov=(
            covariate_data,
            COVARIATE_COL_DICT['COL_DATE'],
            COVARIATE_COL_DICT['COL_LOC_ID']
        ),
        df_beta=mr.get_beta_ode_fit(),
        covmodel_set=all_covmodels_set,
    )
    if regression_specification.data.coefficient_version is not None:
        # If you want to use a specific coefficient version,
        # this will read in the coefficients and then they will be
        # passed to the beta regression.
        coefficient_data = RegressionData(
            regression_dir=Path(regression_specification.data.coefficient_version)
        )

        fixed_coefficients = coefficient_data.load_mr_coefficients(
            draw_id=draw_id
        )
    else:
        fixed_coefficients = None
    # Fit the beta regression; the `path` argument automatically saves the coefficients
    # to the specified file
    mr.fit_beta_regression_prod(
        ordered_covmodel_sets=ordered_covmodel_sets,
        mr_data=mr_data,
        path=regression_data.get_draw_coefficient_file(draw_id),
        df_cov_coef=fixed_coefficients,
        add_intercept=False,
    )
    # -------------------- POST PROCESSING AND SAVING ------------------------ #
    # Get the fitted values of beta from the regression model and append on
    # to the fits -- **this is just for diagnostic purposes**
    regression_fit = regression_data.load_mr_coefficients(
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
        [cov.name for cov in regression_specification.covariates] + ['beta_pred']
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
        loc_beta_fits.to_csv(regression_data.get_draw_beta_fit_file(l_id, draw_id),
                             index=False)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    mr.get_beta_ode_params().to_csv(regression_data.get_draw_beta_param_file(draw_id),
                                    index=False)


def main():

    args = parse_arguments()
    run_beta_regression(draw_id=args.draw_id, regression_version=args.regression_version)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import logging

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.versioner import COVARIATE_COL_DICT
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.versioner import Directories

from seiir_model_pipeline.core.data import load_mr_coefficients
from seiir_model_pipeline.core.data import load_covariates
from seiir_model_pipeline.core.data import load_ode_fits

from seiir_model_pipeline.core.utils import load_locations

from seiir_model_pipeline.core.model_inputs import convert_inputs_for_beta_model
from seiir_model_pipeline.core.model_inputs import convert_to_covmodel

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
    settings = load_regression_settings(regression_version)

    # Load metadata
    directories = Directories(
        regression_version=regression_version,
    )
    # Load locations
    location_ids = load_locations(directories)

    # Load covariates
    covariate_data = load_covariates(
        directories,
        covariate_version=settings.covariate_version,
        location_ids=location_ids
    )
    # Load previous beta ode fits
    df_beta = load_ode_fits(
        directories,
        location_ids=location_ids,
        draw_id=draw_id
    )
    # -------------- BETA REGRESSION WITH LOADED COVARIATES -------------------- #
    # Convert inputs for beta regression using model_inputs utilities functions
    mr = ModelRunner()

    ordered_covmodel_sets, all_covmodels_set = convert_to_covmodel(
        cov_dict=settings.covariates,
        cov_order_list=settings.covariates_order
    )
    mr_data = convert_inputs_for_beta_model(
        data_cov=(
            covariate_data, COVARIATE_COL_DICT['COL_DATE'],
            COVARIATE_COL_DICT['COL_LOC_ID']
        ),
        df_beta=df_beta,
        covmodel_set=all_covmodels_set,
    )
    if settings.coefficient_version is not None:
        # If you want to use a specific coefficient version,
        # this will read in the coefficients and then they will be
        # passed to the beta regression.
        fixed_coefficients = load_mr_coefficients(
            directories=directories,
            draw_id=draw_id,
            regression_version=settings.coefficient_version
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
        add_intercept=False,
    )

    # TODO: Do we overwrite the beta fit files with the fitted
    #  beta or create new files here?
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
    regression_betas = forecasts[
        [COVARIATE_COL_DICT['COL_LOC_ID'], COVARIATE_COL_DICT['COL_DATE']] +
        list(settings.covariates.keys()) + ['beta_pred']
    ]

    for l_id in location_ids:
        loc_beta_fits = regression_betas.loc[regression_betas[COVARIATE_COL_DICT['COL_LOC_ID']] == l_id].copy()
        loc_beta_fits.to_csv(directories.get_draw_beta_regression_file(l_id, draw_id), index=False)


def main():

    args = parse_arguments()
    run_beta_regression(draw_id=args.draw_id, regression_version=args.regression_version)


if __name__ == '__main__':
    main()

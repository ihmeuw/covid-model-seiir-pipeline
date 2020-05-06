from argparse import ArgumentParser
import logging
import numpy as np

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import COVARIATE_COL_DICT, INFECTION_COL_DICT
from seiir_model_pipeline.core.data import load_all_location_data
from seiir_model_pipeline.core.data import load_mr_coefficients
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.utils import convert_to_covmodel
from seiir_model_pipeline.core.data import load_covariates
from seiir_model_pipeline.core.utils import get_locations
from seiir_model_pipeline.core.utils import process_ode_process_input
from seiir_model_pipeline.core.utils import convert_inputs_for_beta_model

log = logging.getLogger(__name__)


def get_args():
    """
    Gets arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    log.info("Initiating SEIIR beta regression.")
    log.info(f"Running for draw {args.draw_id}.")
    log.info(f"This will be regression version {args.regression_version}.")
    # Load metadata
    args.forecast_version = None
    directories = args_to_directories(args)
    settings = load_regression_settings(args.regression_version)
    # Load data
    location_ids = get_locations(
        directories,
        location_set_version_id=settings.location_set_version_id,
        infection_version=settings.infection_version,
        covariate_version=settings.covariate_version
    )
    location_data = load_all_location_data(
        directories=directories, location_ids=location_ids, draw_id=args.draw_id
    )
    covariate_data = load_covariates(
        directories,
        covariate_version=settings.covariate_version,
        location_ids=location_ids
    )
    np.random.seed(args.draw_id)
    beta_fit_inputs = process_ode_process_input(
        settings=settings,
        location_data=location_data,
    )
    mr = ModelRunner()
    mr.fit_beta_ode(beta_fit_inputs)

    # Convert inputs for regression
    ordered_covmodel_sets, all_covmodels_set = convert_to_covmodel(settings.covariates, settings.covariates_order)
    mr_data = convert_inputs_for_beta_model(
        data_cov=(
            covariate_data, COVARIATE_COL_DICT['COL_DATE'],
            COVARIATE_COL_DICT['COL_LOC_ID']
        ),
        df_beta=mr.get_beta_ode_fit(),
        covmodel_set=all_covmodels_set,
    )
    mr.fit_beta_regression_prod(
        ordered_covmodel_sets=ordered_covmodel_sets,
        mr_data=mr_data,
        path=directories.get_draw_coefficient_file(args.draw_id),
    )
    # Get the fitted values of beta from the regression model and append on
    # to the fits
    regression_fit = load_mr_coefficients(
        directories=directories,
        draw_id=args.draw_id
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
    for l_id in location_ids:
        loc_beta_fits = beta_fit_covariates.loc[beta_fit_covariates[INFECTION_COL_DICT['COL_LOC_ID']] == l_id].copy()
        loc_beta_fits.to_csv(directories.get_draw_beta_fit_file(l_id, args.draw_id), index=False)
    mr.get_beta_ode_params().to_csv(directories.get_draw_beta_param_file(args.draw_id), index=False)


if __name__ == '__main__':
    main()

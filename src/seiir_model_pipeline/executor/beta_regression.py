from argparse import ArgumentParser
import logging
import numpy as np

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import PEAK_DATE_FILE, PEAK_DATE_COL_DICT, COVARIATE_COL_DICT, OBSERVED_DICT
from seiir_model_pipeline.core.data import load_all_location_data
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.utils import convert_to_covmodel
from seiir_model_pipeline.core.data import load_covariates, load_peaked_dates
from seiir_model_pipeline.core.utils import get_locations
from seiir_model_pipeline.core.utils import process_ode_process_input

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
        location_metadata_file=directories.get_location_metadata_file(
            location_set_version_id=settings.location_set_version_id
        )
    )
    location_data = load_all_location_data(
        directories=directories, location_ids=location_ids, draw_id=args.draw_id
    )
    peak_data = load_peaked_dates(
        filepath=PEAK_DATE_FILE,
        col_loc_id=PEAK_DATE_COL_DICT['COL_LOC_ID'],
        col_date=PEAK_DATE_COL_DICT['COL_DATE']
    )
    covariate_data = load_covariates(
        directories,
        location_id=location_ids,
        col_loc_id=COVARIATE_COL_DICT['COL_LOC_ID'],
        col_observed=COVARIATE_COL_DICT['COL_OBSERVED']
    )
    # cov_model_set = convert_to_covmodel(settings.covariates)
    np.random.seed(args.draw_id)
    beta_fit_inputs = process_ode_process_input(
        settings=settings,
        location_data=location_data,
        peak_data=peak_data
    )
    mr = ModelRunner()
    mr.fit_beta_ode(beta_fit_inputs)
    mr.save_beta_ode_result(
        fit_file=directories.draw_ode_fit_file(args.draw_id),
        params_file=directories.draw_ode_param_file(args.draw_id)
    )
    # mr.prep_for_regression()
    # mr.regress(
    #     cov_model_set=cov_model_set
    # )
    # coefficients = mr.get_regression_outputs(directories)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser
import logging
import numpy as np

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.file_master import args_to_directories
from seiir_model_pipeline.core.data import load_all_location_data
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.file_master import INFECTION_COL_DICT
from seiir_model_pipeline.core.utils import get_peaked_dates_from_file, sample_params_from_bounds
from seiir_model_pipeline.core.utils import get_cov_model_set_from_settings
from seiir_model_pipeline.core.utils import load_covariates

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
    directories = args_to_directories(args)
    settings = load_regression_settings(directories)

    # Load data
    location_data = load_all_location_data(directories, location_ids=settings['location_ids'])
    peaked_dates = get_peaked_dates_from_file()
    covariate_data = load_observed_covariates(directories, location_ids=settings['location_ids'])

    # Sample the alphas, sigma, gamma1, gamma2
    np.seed(args.draw_id)
    alpha = sample_params_from_bounds(settings['alpha'])
    sigma = sample_params_from_bounds(settings['beta'])
    gamma1 = sample_params_from_bounds(settings['gamma1'])
    gamma2 = sample_params_from_bounds(settings['gamma2'])

    cov_model_set = get_cov_model_set_from_settings(settings['covariates'])

    mr = ModelRunner()
    mr.fit_betas(
        column_dict=INFECTION_COL_DICT,
        df=location_data,
        alpha=settings['alpha'],
        sigma=settings['sigma'],
        gamma1=settings['gamma1'],
        gamma2=settings['gamma2'],
        solver_dt=settings['solver_dt'],
        spline_options={
            'knots': settings['knots'],
            'degree': settings['degree']
        },
        peaked_dates=peaked_dates
    )
    mr.prep_for_regression()
    mr.regress(
        cov_model_set=cov_model_set
    )
    coefficients = mr.get_regression_outputs(directories)


if __name__ == '__main__':
    main()

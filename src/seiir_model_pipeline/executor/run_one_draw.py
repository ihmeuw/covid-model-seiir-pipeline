from argparse import ArgumentParser
import logging
import numpy as np

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.file_master import args_to_directories
from seiir_model_pipeline.core.data import load_all_location_data
from seiir_model_pipeline.core.versioner import load_settings
from seiir_model_pipeline.core.file_master import INFECTION_COL_DICT
from seiir_model_pipeline.core.utils import get_peaked_dates_from_file
from seiir_model_pipeline.core.utils import sample_params_from_bounds
from seiir_model_pipeline.core.utils import get_cov_model_set_from_settings

log = logging.getLogger(__name__)

LOCATION_IDS = [1]  # Where are we going to pull this list? Hierarchies?
# call to db_queries.get_location_metadata(location_set_version_id=...)


def get_args():
    """
    Gets arguments from the command line for only one run.
    """
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--output-version", type=str, required=True)
    parser.add_argument("--warm-start", action='store_true', required=False)

    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR modeling pipeline.")
    log.info(f"Running for draw {args.draw_id}.")
    log.info(f"This will be output version {args.output_version}.")
    if args.warm_start:
        log.info("Will resume from after beta regression.")

    directories = args_to_directories(args)
    settings = load_settings(args.output_version)

    location_data = load_all_location_data(directories, location_ids=LOCATION_IDS)
    peaked_dates = get_peaked_dates_from_file()

    # Sample the alphas, sigma, gamma1, gamma2
    np.seed(args.draw_id)
    alpha = sample_params_from_bounds(settings['alpha'])
    sigma = sample_params_from_bounds(settings['beta'])

    cov_model_set = get_cov_model_set_from_settings(settings['covariates'])

    mr = ModelRunner()
    if not args.warm_start:
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
        mr.save_regression_outputs(directories)
    if args.warm_start:
        mr.load_regression_outputs(directories)
    mr.predict()
    mr.ode()


if __name__ == '__main__':
    main()

from argparse import ArgumentParser
import logging

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.data import cache_covariates
from seiir_model_pipeline.core.versioner import COVARIATE_COL_DICT
from seiir_model_pipeline.core.workflow import SEIIRWorkFlow
from seiir_model_pipeline.core.utils import get_locations

log = logging.getLogger(__name__)


def get_args():
    """
    Get arguments from the command line for this whole fun.
    """
    parser = ArgumentParser()
    parser.add_argument("--regression-version", type=str, required=False, default=None)
    parser.add_argument("--forecast-version", type=str, required=False, default=None)

    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR modeling pipeline.")

    directories = args_to_directories(args)
    directories.make_dirs()
    
    run_regression = args.regression_version is not None
    run_forecasts = args.forecast_version is not None

    wf = SEIIRWorkFlow(directories=directories)

    # TODO: this will need to be more complex when we do scenarios and when we run forecasts
    #  only rather than re-running the regression.
    if run_regression:
        regression_settings = load_regression_settings(args.regression_version)
        location_ids = get_locations(
            directories, regression_settings.location_set_version_id
        )
        cache_covariates(
            directories=directories,
            covariate_names=list(regression_settings.covariates.keys()),
            location_id=location_ids,
            col_loc_id=COVARIATE_COL_DICT['COL_LOC_ID'],
            col_date=COVARIATE_COL_DICT['COL_DATE'],
            col_observed=COVARIATE_COL_DICT['COL_OBSERVED']
        )
        regression_tasks = wf.attach_regression_tasks(
            n_draws=regression_settings.n_draws,
            regression_version=args.regression_version
        )
        if run_forecasts:
            wf.attach_forecast_tasks(
                location_ids=location_ids,
                regression_version=args.regression_version,
                forecast_version=args.forecast_version,
                upstream_tasks=regression_tasks
            )
        wf.run()


if __name__ == '__main__':
    main()

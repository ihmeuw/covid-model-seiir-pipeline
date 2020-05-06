from argparse import ArgumentParser
import logging

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_regression_settings, load_forecast_settings
from seiir_model_pipeline.core.data import cache_covariates
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
    parser.add_argument("--run-splicer", action='store_true', required=False, default=False)
    parser.add_argument("--create-diagnostics", action='store_true', required=False, default=False)

    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR modeling pipeline.")

    directories = args_to_directories(args)
    directories.make_dirs()

    wf = SEIIRWorkFlow(directories=directories)
    
    run_regression = args.regression_version is not None
    run_forecasts = args.forecast_version is not None

    regression_tasks = []

    if run_regression:
        regression_settings = load_regression_settings(args.regression_version)
        location_ids = get_locations(
            directories, regression_settings.location_set_version_id
        )
        cache_covariates(
            directories=directories,
            covariate_versions=regression_settings.covariate_version,
            location_ids=location_ids,
            covariate_draw_dict=regression_settings.covariate_draw_dict
        )
        regression_tasks = wf.attach_regression_tasks(
            n_draws=regression_settings.n_draws,
            regression_version=args.regression_version,
            add_diagnostic=args.create_diagnostics
        )
    else:
        if not run_forecasts:
            raise RuntimeError("You have to run either a regression or forecast.")
        regression_settings = load_regression_settings(args.forecast_version)

    if run_forecasts:
        forecast_settings = load_forecast_settings(args.forecast_version)
        location_ids = get_locations(
            directories, regression_settings.location_set_version_id
        )
        cache_covariates(
            directories=directories,
            covariate_versions=forecast_settings.covariate_version,
            location_ids=location_ids,
            covariate_draw_dict=forecast_settings.covariate_draw_dict
        )
        wf.attach_forecast_tasks(
            location_ids=location_ids,
            add_splicer=args.run_splicer,
            add_diagnostic=args.create_diagnostics,
            regression_version=args.regression_version,
            forecast_version=args.forecast_version,
            upstream_tasks=regression_tasks
        )

    wf.run()


if __name__ == '__main__':
    main()

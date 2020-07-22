from argparse import ArgumentParser
import logging

from covid_model_seiir_pipeline.core.versioner import args_to_directories
from covid_model_seiir_pipeline.core.versioner import load_regression_settings, load_forecast_settings
from covid_model_seiir_pipeline.core.workflow import SEIIRWorkFlow
from covid_model_seiir_pipeline.core.utils import load_locations

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

    run_regression = args.regression_version is not None
    run_forecasts = args.forecast_version is not None

    # Fix for when only forecasting is being run
    #import pdb ; pdb.set_trace()    
    if (args.regression_version is None) and (args.forecast_version is not None):
        forecast_settings = load_forecast_settings(args.forecast_version)
        args.regression_version = forecast_settings.regression_version
    directories = args_to_directories(args)
    directories.make_dirs()

    wf = SEIIRWorkFlow(directories=directories)

    regression_tasks = []

    if run_regression:
        regression_settings = load_regression_settings(args.regression_version)
        regression_tasks = wf.attach_regression_tasks(
            n_draws=regression_settings.n_draws,
            regression_version=args.regression_version,
            add_diagnostic=args.create_diagnostics,
        )
    else:
        if not run_forecasts:
            raise RuntimeError("You have to run either a regression or forecast.")

    if run_forecasts:
        location_ids = load_locations(directories)
        forecast_settings = load_forecast_settings(args.forecast_version)
        wf.attach_forecast_tasks(
            location_ids=location_ids,
            add_splicer=args.run_splicer,
            add_diagnostic=args.create_diagnostics,
            regression_version=forecast_settings.regression_version,
            forecast_version=args.forecast_version,
            upstream_tasks=regression_tasks
        )

    wf.run()


if __name__ == '__main__':
    main()

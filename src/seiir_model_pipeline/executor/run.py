from argparse import ArgumentParser
import logging

from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_ode_settings, load_regression_settings, load_forecast_settings
from seiir_model_pipeline.core.workflow import SEIIRWorkFlow
from seiir_model_pipeline.core.utils import load_locations

log = logging.getLogger(__name__)


def get_args():
    """
    Get arguments from the command line for this whole fun.
    """
    parser = ArgumentParser()
    parser.add_argument("--ode-version", type=str, required=False, default=None)
    parser.add_argument("--regression-version", type=str, required=False, default=None)
    parser.add_argument("--forecast-version", type=str, required=False, default=None)
    parser.add_argument("--run-splicer", action='store_true', required=False, default=False)
    parser.add_argument("--create-diagnostics", action='store_true', required=False, default=False)

    return parser.parse_args()


def main():
    args = get_args()
    log.info("Initiating SEIIR modeling pipeline.")

    directories = args_to_directories(args)

    wf = SEIIRWorkFlow(directories=directories)

    run_ode = args.ode_version is not None
    run_regression = args.regression_version is not None
    run_forecasts = args.forecast_version is not None

    ode_tasks = []
    regression_tasks = []

    if run_ode:
        ode_settings = load_ode_settings(args.ode_version)
        ode_tasks = wf.attach_ode_tasks(
            n_draws=ode_settings.n_draws,
            ode_version=args.ode_version
        )

    if run_regression:
        regression_settings = load_regression_settings(args.regression_version)
        ode_settings = load_ode_settings(regression_settings.ode_version)

        regression_tasks = wf.attach_regression_tasks(
            n_draws=ode_settings.n_draws,
            regression_version=args.regression_version,
            add_diagnostic=args.create_diagnostics,
            upstream_tasks=ode_tasks
        )
    else:
        if not run_forecasts:
            raise RuntimeError("You have to run either a regression or forecast.")

    if run_forecasts:
        location_ids = load_locations(directories)
        wf.attach_forecast_tasks(
            location_ids=location_ids,
            add_splicer=args.run_splicer,
            add_diagnostic=args.create_diagnostics,
            forecast_version=args.forecast_version,
            upstream_tasks=regression_tasks
        )

    wf.run()


if __name__ == '__main__':
    main()

from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import logging

from seiir_model_pipeline.core.versioner import Directories
from seiir_model_pipeline.core.versioner import load_ode_settings, load_regression_settings, load_forecast_settings
from seiir_model_pipeline.core.workflow import SEIIRWorkFlow
from seiir_model_pipeline.core.utils import load_locations

log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Get arguments from the command line for this whole fun.
    """
    parser = ArgumentParser()
    parser.add_argument("--ode-version", type=str, required=False, default=None)
    parser.add_argument("--regression-version", type=str, required=False, default=None)
    parser.add_argument("--forecast-version", type=str, required=False, default=None)
    parser.add_argument("--run-splicer", action='store_true', required=False, default=False)
    parser.add_argument("--create-diagnostics", action='store_true', required=False, default=False)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def run(ode_version: str, regression_version: str, forecast_version: str,
        run_splicer: bool, create_diagnostics: bool):

    log.info("Initiating SEIIR modeling pipeline.")

    directories = Directories(
        ode_version=ode_version,
        regression_version=regression_version,
        forecast_version=forecast_version
    )
    wf = SEIIRWorkFlow(directories=directories)

    run_ode = ode_version is not None
    run_regression = regression_version is not None
    run_forecasts = forecast_version is not None

    ode_tasks = []
    regression_tasks = []

    if run_ode:
        ode_settings = load_ode_settings(ode_version)
        ode_tasks = wf.attach_ode_tasks(
            n_draws=ode_settings.n_draws,
            ode_version=ode_version
        )

    if run_regression:
        regression_settings = load_regression_settings(regression_version)
        ode_settings = load_ode_settings(regression_settings.ode_version)

        regression_tasks = wf.attach_regression_tasks(
            n_draws=ode_settings.n_draws,
            regression_version=regression_version,
            add_diagnostic=create_diagnostics,
            upstream_tasks=ode_tasks
        )
    else:
        if not run_forecasts:
            raise RuntimeError("You have to run either a regression or forecast.")

    if run_forecasts:
        location_ids = load_locations(directories)
        wf.attach_forecast_tasks(
            location_ids=location_ids,
            add_splicer=run_splicer,
            add_diagnostic=create_diagnostics,
            forecast_version=forecast_version,
            upstream_tasks=regression_tasks
        )

    wf.run()


def main():

    args = parse_arguments()
    run(ode_version=args.ode_version, regression_version=args.regression_version,
        forecast_version=args.forecast_version, run_splicer=args.run_splicer,
        create_diagnostics=args.create_diagnostics)


if __name__ == '__main__':
    main()

from bdb import BdbQuit
import functools
from typing import Any, Callable

import click
# This is just exposing the api from this namespace.
from covid_shared.cli_tools import (
    add_verbose_and_with_debugger,
    configure_logging_to_terminal,
)

with_regression_version = click.option(
    '--regression-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"regression_specification.yaml".'
)

with_forecast_version = click.option(
    '--forecast-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"forecast_specification.yaml".'
)

with_postprocessing_version = click.option(
    '--postprocessing-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"postprocessing_specification.yaml".'
)

with_scenario = click.option(
    '--scenario', '-s',
    type=click.STRING,
    required=True,
    help='The scenario to be run.'
)

with_measure = click.option(
    '--measure', '-m',
    type=click.STRING,
    required=True,
    help='The measure to be run.'
)

with_draw_id = click.option(
    '--draw-id', '-d',
    type=click.INT,
    required=True,
    help='The draw to be run.'
)


def handle_exceptions(func: Callable, logger: Any, with_debugger: bool) -> Callable:
    """Drops a user into an interactive debugger if func raises an error."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (BdbQuit, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.exception("Uncaught exception {}".format(e))
            if with_debugger:
                import pdb
                import traceback
                traceback.print_exc()
                pdb.post_mortem()
            else:
                raise

    return wrapped

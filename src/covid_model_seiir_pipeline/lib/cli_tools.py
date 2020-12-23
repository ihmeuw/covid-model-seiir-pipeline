from bdb import BdbQuit
from collections import defaultdict
import functools
import time
from typing import Any, Callable

import click
# This is just exposing the api from this namespace.
from covid_shared.cli_tools import (
    add_verbose_and_with_debugger,
    configure_logging_to_terminal,
)
from loguru import logger


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


class _TaskPerformanceLogger:

    def __init__(self):
        self.current_context = None
        self.current_context_start = None
        self.times = defaultdict(float)

    def _record_timing(self, context):
        if context is None:
            return
        if self.current_context is None:
            self.current_context = context
            self.current_context_start = time.time()
        else:
            self.times[self.current_context] += time.time() - self.current_context_start
            self.current_context = context
            self.current_context_start = time.time()

    def info(self, *args, context=None, **kwargs):
        self._record_timing(context)
        logger.info(*args, **kwargs)

    def debug(self, *args, context=None, **kwargs):
        self._record_timing(context)
        logger.debug(*args, **kwargs)

    def warning(self, *args, context=None, **kwargs):
        self._record_timing(context)
        logger.debug(*args, **kwargs)

    def report(self):
        if self.current_context is not None:
            self.times[self.current_context] += time.time() - self.current_context_start
        logger.info(
            "\nRuntime report\n" +
            "="*31 + "\n" +
            "\n".join([f'{context:<20}:{elapsed_time:>10.2f}' for context, elapsed_time in self.times.items()])
        )


task_performance_logger = _TaskPerformanceLogger()

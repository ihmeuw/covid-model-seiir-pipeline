import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_beta_resampling(fit_version: str, progress_bar: bool) -> None:
    logger.info(f'Starting beta_resampling.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_threads = specification.workflow.task_specifications[FIT_JOBS.beta_resampling].num_cores

    for measure in ['case', 'admission', 'death']:
        pass

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_resampling(fit_version: str,
                    progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_resampling, logger, with_debugger)
    run(fit_version=fit_version,
        progress_bar=progress_bar)

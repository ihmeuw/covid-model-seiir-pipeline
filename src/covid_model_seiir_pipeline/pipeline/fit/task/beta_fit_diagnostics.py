import functools
from pathlib import Path
import tempfile
from typing import Dict
import warnings

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    parallel,
    pdf_merger,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit.model import plotter

logger = cli_tools.task_performance_logger


def run_beta_fit_diagnostics(fit_version: str, progress_bar) -> None:
    logger.info('Starting beta fit.', context='setup')
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_cores = specification.workflow.task_specifications[FIT_JOBS.beta_fit_diagnostics].num_cores

    logger.info('Loading beta fit summary data', context='read')

    hierarchy = data_interface.load_hierarchy('pred')
    name_map = hierarchy.set_index('location_id').location_name
    deaths = data_interface.load_summary('daily_deaths')
    locs = deaths.location_id.unique()
    # These have a standard index, so we're not clipping to any location.
    start, end = deaths.date.min(), deaths.date.max()
    locations = [plotter.Location(loc_id, name_map.loc[loc_id]) for loc_id in locs]
    logger.info('Building location specific plots', context='make_plots')
    with tempfile.TemporaryDirectory() as temp_dir_name:
        plot_cache = Path(temp_dir_name)

        _runner = functools.partial(
            plotter.ies_plot,
            data_interface=data_interface,
            start=start,
            end=end,
            review=False,
            plot_root=plot_cache,
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parallel.run_parallel(
                runner=_runner,
                arg_list=locations,
                num_cores=num_cores,
                progress_bar=progress_bar,
            )

        logger.info('Collating plots', context='merge_plots')
        output_path = Path(specification.data.output_root) / f'past_infections.pdf'
        pdf_merger.merge_pdfs(
            plot_cache=plot_cache,
            output_path=output_path,
            patterns=['ies_{location_id}'],
            hierarchy=hierarchy,
        )

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def beta_fit_diagnostics(fit_version: str,
                         progress_bar: bool,
                         verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_fit_diagnostics, logger, with_debugger)
    run(fit_version=fit_version,
        progress_bar=progress_bar)

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
    data_dict = build_data_dict(
        data_interface=data_interface,
        round_id=2,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    hierarchy = data_interface.load_hierarchy('pred')
    name_map = hierarchy.set_index('location_id').location_name
    deaths = data_dict['daily_deaths'].reset_index()
    locs = deaths.location_id.unique()
    # These have a standard index, so we're not clipping to any location.
    start, end = deaths.date.min(), deaths.date.max()
    locations = [plotter.Location(loc_id, name_map.loc[loc_id]) for loc_id in locs]
    data_dicts = []
    for location_id in locs:
        loc_data_dict = {}
        for measure, data in data_dict.items():
            try:
                loc_data_dict[measure] = data.loc[location_id]
            except KeyError:
                loc_data_dict[measure] = data
        data_dicts.append((
            plotter.Location(location_id, name_map.loc[location_id]),
            loc_data_dict,
        ))

    logger.info('Building location specific plots', context='make_plots')
    with tempfile.TemporaryDirectory() as temp_dir_name:
        plot_cache = Path(temp_dir_name)

        _runner = functools.partial(
            plotter.ies_plot,
            data_dictionary=data_dict,
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


def load_and_select_round(measure: str, round_id: int, data_interface: FitDataInterface) -> pd.DataFrame:
    data = data_interface.load_summary(measure)
    index_names = set(data.index.names)
    final_index_names = [n for n in ['measure', 'location_id', 'date'] if n in index_names]
    if 'round' in index_names:
        data = data.reset_index().set_index(['round'] + final_index_names).sort_index().loc[round_id]
    return data


def build_data_dict(data_interface: FitDataInterface,
                    round_id: int,
                    num_cores: int,
                    progress_bar: bool) -> Dict[str, pd.DataFrame]:
    data_dict = data_interface.load_summary('data_dictionary')

    _runner = functools.partial(
        load_and_select_round,
        round_id=round_id,
        data_interface=data_interface,
    )
    measures = data_dict.output.tolist()

    data = parallel.run_parallel(
        runner=_runner,
        arg_list=measures,
        num_cores=num_cores,
        progress_bar=progress_bar
    )

    data = dict(zip(measures, data))
    return data


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

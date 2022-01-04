import functools
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple
import warnings

import click
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    parallel,
    pdf_merger,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit.model import plotter

logger = cli_tools.task_performance_logger


def run_beta_fit_diagnostics(fit_version: str, plot_type: str, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_cores = specification.workflow.task_specifications[FIT_JOBS.beta_fit_diagnostics].num_cores

    if plot_type == plotter.PLOT_TYPE.model_fit:
        output_root = Path(fit_version)

        version_map = {'reference': (data_interface, 2)}
        plot_function = plotter.model_fit_plot
        output_path = output_root / f'past_infections_{output_root.name}.pdf'
        patterns = ['ies_{location_id}']
    elif plot_type == plotter.PLOT_TYPE.model_compare:
        output_root = Path(fit_version)
        new_version = output_root.name
        old_spec = FitSpecification.from_version_root(specification.data.compare_version)
        old_data_interface = FitDataInterface.from_specification(old_spec)
        old_version = Path(specification.data.compare_version).name
        version_map = {
            f'{new_version} Round 1': (data_interface, 1),
            f'{new_version} Round 2': (data_interface, 2),
            f'{old_version} Round 1': (old_data_interface, 1),
            f'{old_version} Round 2': (old_data_interface, 2),
        }
        plot_function = plotter.model_compare_plot
        output_path = output_root / f'infections_compare_{output_root.name}.pdf'
        patterns = ['compare_{location_id}']
    else:
        raise ValueError(f'Unknown plot type {plot_type}.')

    logger.info('Loading and configuring plot locations and dates.', context='read')
    hierarchy = data_interface.load_hierarchy('pred')
    name_map = hierarchy.set_index('location_id').location_name
    deaths = data_interface.load_summary('daily_deaths').reset_index()
    # These have a standard index, so we're not clipping to any location.
    start, end = deaths.date.min(), deaths.date.max()

    logger.info('Loading beta fit summary data', context='read')
    data_dicts = {
        version_name: build_data_dict(
            data_interface=version_data_interface,
            round_id=version_round_id,
            num_cores=num_cores,
            progress_bar=progress_bar,
        ) for version_name, (version_data_interface, version_round_id) in version_map
    }

    logger.info('Partitioning data dictionary by location', context='partition')
    data_dicts = partition_data_dict(
        data_dicts=data_dicts,
        locations=deaths.location_id.unique(),
        name_map=name_map,
        progress_bar=progress_bar
    )

    logger.info('Building location specific plots', context='make_plots')
    with tempfile.TemporaryDirectory() as temp_dir_name:
        plot_cache = Path(temp_dir_name)

        _runner = functools.partial(
            plot_function,
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
                arg_list=data_dicts,
                num_cores=num_cores,
                progress_bar=progress_bar,
            )

        logger.info('Collating plots', context='merge_plots')
        pdf_merger.merge_pdfs(
            plot_cache=plot_cache,
            output_path=output_path,
            patterns=patterns,
            hierarchy=hierarchy,
        )

    logger.report()


def load_and_select_round(measure: str, round_id: int, data_interface: FitDataInterface) -> pd.DataFrame:
    data = data_interface.load_summary(measure)
    index_names = set(data.index.names)
    final_index_names = [n for n in ['location_id', 'measure', 'date'] if n in index_names]
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


def partition_data_dict(data_dicts: plotter.DataDict,
                        locations: List[int],
                        name_map: pd.Series,
                        progress_bar: bool) -> List[Tuple[plotter.Location, plotter.DataDict]]:
    loc_data_dicts = []
    for location_id in tqdm.tqdm(locations, disable=not progress_bar):
        loc_data_dict = {v: {} for v in data_dicts}
        for version, dd in data_dicts.items():
            for measure, data in dd.items():
                try:
                    loc_data_dict[version][measure] = data.loc[location_id]
                except KeyError:
                    loc_data_dict[version][measure] = None
        loc_data_dicts.append((
            plotter.Location(location_id, name_map.loc[location_id]),
            loc_data_dict,
        ))
    return loc_data_dicts


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

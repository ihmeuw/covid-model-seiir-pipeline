import functools
import multiprocessing
from pathlib import Path
import tempfile

import click
from loguru import logger
import pandas as pd

import tqdm

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import (
    DiagnosticsSpecification,
)
from covid_model_seiir_pipeline.pipeline.diagnostics import model


def run_grid_plots(diagnostics_version: str, name: str) -> None:
    logger.info(f'Starting grid plots for version {diagnostics_version}, name {name}.')
    diagnostics_spec = DiagnosticsSpecification.from_path(
        Path(diagnostics_version) / static_vars.DIAGNOSTICS_SPECIFICATION_FILE
    )
    grid_plot_spec = [spec for spec in diagnostics_spec.grid_plots if spec.name == name].pop()
    logger.info('Building plot versions')
    plot_versions = model.make_plot_versions(grid_plot_spec.comparators, model.COLOR_MAP)

    cache_draws = ['beta_scaling_parameters', 'coefficients']

    with tempfile.TemporaryDirectory() as temp_dir_name:
        root = Path(temp_dir_name)
        logger.info('Building data cache')
        data_cache = root / 'data_cache'
        data_cache.mkdir()
        for plot_version in plot_versions:
            plot_version.build_cache(data_cache, cache_draws)

        plot_cache = root / 'plot_cache'
        plot_cache.mkdir()

        logger.info('Loading locations')
        hierarchy = plot_versions[0].load_output_miscellaneous('hierarchy', is_table=True)
        deaths = plot_versions[0].load_output_summaries('daily_deaths')
        modeled_locs = hierarchy.loc[hierarchy.location_id.isin(deaths.location_id.unique()),
                                     ['location_id', 'location_name']]
        locs_to_plot = [model.Location(loc[1], loc[2]) for loc in modeled_locs.itertuples()]

        _runner = functools.partial(
            model.make_grid_plot,
            plot_versions=plot_versions,
            date_start=pd.to_datetime(grid_plot_spec.date_start),
            date_end=pd.to_datetime(grid_plot_spec.date_end),
            output_dir=plot_cache,
        )

        num_cores = diagnostics_spec.workflow.task_specifications['grid_plots'].num_cores
        logger.info('Starting plots')
        with multiprocessing.Pool(num_cores) as pool:
            list(tqdm.tqdm(pool.imap(_runner, locs_to_plot), total=len(locs_to_plot)))

        logger.info('Collating plots')
        output_path = Path(diagnostics_spec.data.output_root) / f'grid_plots_{grid_plot_spec.name}.pdf'
        model.merge_pdfs(plot_cache, output_path, [loc.id for loc in locs_to_plot], hierarchy)


@click.command()
@cli_tools.with_diagnostics_version
@cli_tools.with_name
@cli_tools.add_verbose_and_with_debugger
def grid_plots(diagnostics_version: str, name: str,
               verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_grid_plots, logger, with_debugger)
    run(diagnostics_version=diagnostics_version,
        name=name,)


if __name__ == '__main__':
    grid_plots()

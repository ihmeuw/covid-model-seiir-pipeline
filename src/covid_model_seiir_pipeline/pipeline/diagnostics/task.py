from pathlib import Path
import tempfile
from typing import List, NamedTuple, Tuple

import click
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import (
    GridPlotsComparatorSpecification,
    DiagnosticsSpecification,
    DIAGNOSTICS_JOBS,
)
from covid_model_seiir_pipeline.pipeline.postprocessing import (
    PostprocessingSpecification,
    PostprocessingDataInterface,
)

sns.set_style('whitegrid')


COLOR_MAP = plt.get_cmap('tab10')


Color = Tuple[float, float, float, float]


class PlotVersion(NamedTuple):
    version: Path
    scenario: str
    label: str
    color: Color


def make_plot_versions(comparators: List[GridPlotsComparatorSpecification]) -> List[PlotVersion]:
    plot_versions = []
    for comparator in comparators:
        for scenario, label in comparator.scenarios.items():
            plot_versions.append((Path(comparator.version), scenario, label))
    plot_versions = [PlotVersion(*pv, COLOR_MAP(i)) for i, pv in enumerate(plot_versions)]
    return plot_versions


def get_pdi(outputs_version: Path) -> PostprocessingDataInterface:
    spec = PostprocessingSpecification.from_path(outputs_version / static_vars.POSTPROCESSING_SPECIFICATION_FILE)
    pdi = PostprocessingDataInterface.from_specification(spec)
    return pdi


def make_time_plot(ax, plot_versions: List[PlotVersion], measure: str, loc_id: int,
                   start: pd.Timestamp, end: pd.Timestamp):
    locator = mdates.AutoDateLocator(maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    for plot_version in plot_versions:
        data = get_pdi(plot_version.version).load_output_summaries(plot_version.scenario, measure)
        data = data[data.location_id == loc_id]
        data['date'] = pd.to_datetime(data['date'])
        ax.plot(data['date'], data['mean'], color=plot_version.color)
        ax.fill_between(data['date'], data['upper'], data['mean'], alpha=0.2, color=plot_version.color)
        ax.fill_between(data['date'], data['lower'], data['mean'], alpha=0.2, color=plot_version.color)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def make_coefficient_plot(ax, plot_versions: List[PlotVersion], covariate: str, loc_id: int):
    for i, plot_version in enumerate(plot_versions):
        data = get_pdi(plot_version.version).load_output_draws(plot_version.scenario, 'coefficients')
        data = data.set_index(['location_id', 'covariate']).loc[(loc_id, covariate)]
        plt.boxplot(data,
                    positions=[i],
                    widths=[.7],
                    boxprops=dict(color=plot_version.color, linewidth=2),
                    capprops=dict(color=plot_version.color, linewidth=2),
                    whiskerprops=dict(color=plot_version.color, linewidth=2),
                    flierprops=dict(color=plot_version.color, markeredgecolor=plot_version.color, linewidth=2),
                    medianprops=dict(color=plot_version.color, linewidth=2), labels=[' '])
    ax.set_ylabel(covariate)


def make_results_page(plot_versions: List[PlotVersion], location_id: int, start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    figure = plt.figure(figsize=(11, 8.5), constrained_layout=True)
    grid_spec = figure.add_gridspec(nrows=2, ncols=4)

    make_time_plot(grid_spec[0, 0], plot_versions, 'r_effective', location_id, start, end)
    grid_spec[0, 0].set_ylim(0, 2)

    make_time_plot(grid_spec[0, 1], plot_versions, 'betas', location_id, start, end)
    make_time_plot(grid_spec[0, 2], plot_versions, 'log_beta_residuals', location_id, start, end)
    make_time_plot(grid_spec[1, 0], plot_versions, 'daily_infections', location_id, start, end)
    make_time_plot(grid_spec[1, 1], plot_versions, 'daily_deaths', location_id, start, end)
    make_time_plot(grid_spec[1, 2], plot_versions, 'cumulative_infections', location_id, start, end)
    make_time_plot(grid_spec[1, 3], plot_versions, 'cumulative_deaths', location_id, start, end)

    figure.suptitle(location_id, fontsize=24)

    if plot_file:
        figure.save_fig(plot_file, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()



def make_covariates_page():
    pass


def make_grid_plot(location_id: int,
                   plot_versions: List[PlotVersion],
                   date_start: pd.Timestamp,
                   date_end: pd.Timestamp,
                   output_dir: str):
    pass


def run_grid_plots(diagnostics_version: str, name: str) -> None:
    logger.info(f'Starting grid plots for version {diagnostics_version}, name {name}.')
    diagnostics_spec = DiagnosticsSpecification.from_path(
        Path(diagnostics_version) / static_vars.DIAGNOSTICS_SPECIFICATION_FILE
    )
    grid_plot_spec = [spec for spec in diagnostics_spec.grid_plots if spec.file_suffix == name].pop()
    plot_versions = make_plot_versions(grid_plot_spec.comparators)

    with tempfile.TemporaryDirectory() as temp_dir_name:
        pass


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

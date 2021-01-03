from pathlib import Path
import tempfile
from typing import List, NamedTuple, Tuple

import click
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np
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


COLOR_MAP = plt.get_cmap('Set1')


Color = Tuple[float, float, float, float]


class PlotVersion:
    
    def __init__(self, 
                 version: Path, 
                 scenario: str,
                 label: str,
                 color: Color):
        self.version = version
        self.scenario = scenario
        self.label = label
        self.color = color
        spec = PostprocessingSpecification.from_path(
            self.version / static_vars.POSTPROCESSING_SPECIFICATION_FILE
        )
        self.pdi = PostprocessingDataInterface.from_specification(spec)
    
    def load_output_summaries(self, measure: str):
        return self.pdi.load_output_summaries(self.scenario, measure)
    
    def load_output_draws(self, measure: str):
        return self.pdi.load_output_draws(self.scenario, measure)
    
    def load_output_miscellaneous(self, measure: str, is_table: bool):
        return self.pdi.load_output_miscellaneous(self.scenario, measure, is_table)


def make_plot_versions(comparators: List[GridPlotsComparatorSpecification]) -> List[PlotVersion]:
    plot_versions = []
    for comparator in comparators:
        for scenario, label in comparator.scenarios.items():
            plot_versions.append((Path(comparator.version), scenario, label))
    plot_versions = [PlotVersion(*pv, COLOR_MAP(i)) for i, pv in enumerate(plot_versions)]
    return plot_versions


def make_time_plot(ax, plot_versions: List[PlotVersion], measure: str, loc_id: int,
                   start: pd.Timestamp, end: pd.Timestamp, vlines=[], label=None, transform=lambda x: x):
    label = label if label else measure
    locator = mdates.AutoDateLocator(maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
    for plot_version in plot_versions:
        data = plot_version.load_output_summaries(measure)
        data = data[data.location_id == loc_id]
        data['date'] = pd.to_datetime(data['date'])
        data[['mean', 'upper', 'lower']] = transform(data[['mean', 'upper', 'lower']])
        ax.plot(data['date'], data['mean'], color=plot_version.color)
        ax.fill_between(data['date'], data['upper'], data['mean'], alpha=0.2, color=plot_version.color)
        ax.fill_between(data['date'], data['lower'], data['mean'], alpha=0.2, color=plot_version.color)
    for vline_x in vlines:
        ax.vlines(vline_x, 0, 1, transform=ax.get_xaxis_transform(), linestyle='dashed', color='grey', alpha=0.8)
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylabel(label, fontsize=14)

    
def make_log_beta_resid_hist(ax, plot_versions: List[PlotVersion], loc_id: int):
    for plot_version in plot_versions:
        data = plot_version.load_output_draws('beta_scaling_parameters')
        data = (data[(data.location_id == loc_id) & (data.scaling_parameter == 'log_beta_residual_mean')]
                .drop(columns=['location_id', 'scaling_parameter']).T)
        ax.hist(data.values, color=plot_version.color, bins=25, 
                histtype='step')
        ax.hist(data.values, color=plot_version.color, bins=25, 
                histtype='stepfilled', alpha=0.2)
    
    ax.set_ylabel('count of log beta residual mean', fontsize=14)

        

def make_coefficient_plot(ax, plot_versions: List[PlotVersion], covariate: str, loc_id: int):
    for i, plot_version in enumerate(plot_versions):
        data = plot_version.load_output_draws('coefficients')
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
    

def make_legend_handles(plot_versions: List[PlotVersion]):
    handles = [mlines.Line2D([], [], color=pv.color, label=pv.label) for pv in plot_versions]
    return handles
        


def make_results_page(plot_versions: List[PlotVersion], location_id: int, start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    fig = plt.figure(figsize=(30, 15), tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=2, ncols=4)
    grid_spec.update(top=0.92, bottom=0.08)

    ax_r_eff = fig.add_subplot(grid_spec[0, 0])
    ax_betas = fig.add_subplot(grid_spec[0, 1])
    ax_resid = fig.add_subplot(grid_spec[0, 2])
    ax_rhist = fig.add_subplot(grid_spec[0, 3])
    
    ax_daily_infec = fig.add_subplot(grid_spec[1, 0])
    ax_daily_death = fig.add_subplot(grid_spec[1, 1])
    ax_cumul_infec = fig.add_subplot(grid_spec[1, 2])
    ax_cumul_death = fig.add_subplot(grid_spec[1, 3])
    
    observed_color = COLOR_MAP(len(plot_versions))
    
    pv = plot_versions[0]
    pop = pv.load_output_miscellaneous('populations', is_table=True)
    pop = pop.loc[(pop.location_id == location_id) &
                  (pop.age_group_id == 22) &
                  (pop.sex_id == 3), 'population'].iloc[0]
    
    full_data = pv.load_output_miscellaneous('full_data_es_processed', is_table=True)
    full_data = full_data[full_data.location_id == location_id]
    full_data['date'] = pd.to_datetime(full_data['date'])
    data_date = full_data.loc[full_data.cumulative_deaths.notnull(), 'date'].max()
    short_term_forecast_date = data_date + pd.Timedelta(days=8)
    vlines = [data_date, short_term_forecast_date]
    
    make_time_plot(ax_r_eff, plot_versions, 'r_effective', location_id, start, end, vlines=vlines)
    ax_r_eff.set_ylim(0, 2)
    make_time_plot(ax_betas, plot_versions, 'betas', location_id, start, end,
                   vlines=vlines,
                   transform=lambda x: np.log(x))
    make_time_plot(ax_resid, plot_versions, 'log_beta_residuals', location_id, start, end, vlines=vlines)
    make_log_beta_resid_hist(ax_rhist, plot_versions, location_id)
    
    make_time_plot(ax_daily_infec, plot_versions, 'daily_infections', location_id, start, end, vlines=vlines)
    ax_daily_infec.plot(full_data['date'], 
                        full_data['cumulative_cases'] - full_data['cumulative_cases'].shift(1),
                        color=observed_color,
                        alpha=0.5)
    make_time_plot(ax_daily_death, plot_versions, 'daily_deaths', location_id, start, end, vlines=vlines)
    ax_daily_death.plot(full_data['date'], 
                        full_data['cumulative_deaths'] - full_data['cumulative_deaths'].shift(1),
                        color=observed_color,
                        alpha=0.5)
    make_time_plot(ax_cumul_infec, plot_versions, 'cumulative_infections', location_id, start, end,
                   vlines=vlines,
                   label='Cumulative infected (% population)',
                   transform=lambda x: 100 * x / pop)
    make_time_plot(ax_cumul_death, plot_versions, 'cumulative_deaths', location_id, start, end, vlines=vlines)

    fig.suptitle(location_id, x=0.15, fontsize=24)
    fig.legend(handles=make_legend_handles(plot_versions), 
               loc='lower center',
               bbox_to_anchor=(0.4, 0),
               fontsize=14,
               frameon=False,
               ncol=len(plot_versions))

    if plot_file:
        fig.save_fig(plot_file, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()



def make_covariates_page():
    fig = plt.figure(figsize=(24, 15), constrained_layout=True)
    grid_spec = fig.add_gridspec(nrows=3, ncols=2, widths=[5, 3])
    
    
    


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
    grid_plot_spec = [spec for spec in diagnostics_spec.grid_plots if spec.name == name].pop()
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

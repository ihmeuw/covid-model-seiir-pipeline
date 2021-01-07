from pathlib import Path
from typing import List
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.pipeline.diagnostics.model.plot_version import (
    Location,
    PlotVersion,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.model import (
    COVARIATES,
)

COLOR_MAP = plt.get_cmap('Set1')
DATE_LOCATOR = mdates.AutoDateLocator(maxticks=10)
DATE_FORMATTER = mdates.ConciseDateFormatter(DATE_LOCATOR, show_offset=False)
FILL_ALPHA = 0.2
OBSERVED_ALPHA = 0.5
AX_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 24
HIST_BINS = 25
FIG_SIZE = (30, 15)
GRID_SPEC_MARGINS = {'top': 0.92, 'bottom': 0.08}
SHORT_RANGE_FORECAST = pd.Timedelta(days=8)


def make_grid_plot(location: Location,
                   plot_versions: List[PlotVersion],
                   date_start: pd.Timestamp, date_end: pd.Timestamp,
                   output_dir: Path):
    """Makes all pages of grid plots from a single location."""
    sns.set_style('whitegrid')
    with warnings.catch_warnings():
        # Suppress some noisy matplotlib warnings.
        warnings.filterwarnings('ignore')
        make_covariates_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_covariates.pdf')
        )
        make_results_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_results.pdf')
        )


def make_covariates_page(plot_versions: List[PlotVersion],
                         location: Location,
                         start: pd.Timestamp, end: pd.Timestamp,
                         plot_file: str = None):
    # FIXME: This is brittle w/r/t new covariates
    time_varying = [c for c, c_config in COVARIATES.items() if c_config.time_varying]
    non_time_varying = [c for c, c_config in COVARIATES.items() if not c_config.time_varying]

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=3, ncols=2, width_ratios=[6, 3], wspace=0.1)
    grid_spec.update(**GRID_SPEC_MARGINS)

    sub_grid_wspace = 0.3
    gs_cov = grid_spec[0, 0].subgridspec(1, len(time_varying), wspace=sub_grid_wspace)
    gs_coef_tv = grid_spec[1, 0].subgridspec(1, len(time_varying), wspace=sub_grid_wspace)
    gs_coef_non_tv = grid_spec[2, :].subgridspec(1, len(non_time_varying), wspace=sub_grid_wspace)
    gs_elastispliner = grid_spec[:2, 1].subgridspec(1, 1)

    for i, covariate in enumerate(time_varying):
        ax_cov = fig.add_subplot(gs_cov[0, i])
        make_time_plot(
            ax_cov,
            plot_versions,
            covariate,
            location.id,
            start, end,
            label=covariate.title(),
        )

        ax_coef = fig.add_subplot(gs_coef_tv[0, i])
        make_coefficient_plot(
            ax_coef,
            plot_versions,
            covariate,
            location.id,
            label=covariate.title(),
        )

    for i, covariate in enumerate(non_time_varying):
        ax_coef = fig.add_subplot(gs_coef_non_tv[0, i])
        make_coefficient_plot(
            ax_coef,
            plot_versions,
            covariate,
            location.id,
            label=covariate.title(),
        )

    ax_es = fig.add_subplot(gs_elastispliner[0, 0])
    # This is an overestimate, but not an important one.
    es_end_date = pd.Timestamp.today() + SHORT_RANGE_FORECAST
    make_time_plot(
        ax_es,
        plot_versions,
        'daily_elastispliner_smoothed',
        location.id,
        start, es_end_date,
        label='Elastispliner',
    )

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_results_page(plot_versions: List[PlotVersion],
                      location: Location,
                      start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    observed_color = COLOR_MAP(len(plot_versions))

    # Load some shared data.
    pv = plot_versions[0]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    full_data = pv.load_output_miscellaneous('full_data_es_processed', is_table=True, location_id=location.id)
    data_date = full_data.loc[full_data.cumulative_deaths.notnull(), 'date'].max()
    short_term_forecast_date = data_date + SHORT_RANGE_FORECAST
    vlines = [data_date, short_term_forecast_date]

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=2, ncols=4)
    grid_spec.update(**GRID_SPEC_MARGINS)

    ax_r_eff = fig.add_subplot(grid_spec[0, 0])
    ax_betas = fig.add_subplot(grid_spec[0, 1])
    ax_resid = fig.add_subplot(grid_spec[0, 2])
    ax_rhist = fig.add_subplot(grid_spec[0, 3])

    ax_daily_infec = fig.add_subplot(grid_spec[1, 0])
    ax_daily_death = fig.add_subplot(grid_spec[1, 1])
    ax_cumul_infec = fig.add_subplot(grid_spec[1, 2])
    ax_cumul_death = fig.add_subplot(grid_spec[1, 3])

    make_time_plot(
        ax_r_eff,
        plot_versions,
        'r_effective',
        location.id,
        start, end,
        label='R Effective',
        vlines=vlines,
    )
    ax_r_eff.set_ylim(0, 2)

    make_time_plot(
        ax_betas,
        plot_versions,
        'betas',
        location.id,
        start, end,
        label='Log Beta',
        vlines=vlines,
        transform=lambda x: np.log(x),
    )

    make_time_plot(
        ax_resid,
        plot_versions,
        'log_beta_residuals',
        location.id,
        start, end,
        label='Log Beta Residuals',
        vlines=vlines,
    )

    make_log_beta_resid_hist(
        ax_rhist,
        plot_versions,
        location.id
    )

    make_time_plot(
        ax_daily_infec,
        plot_versions,
        'daily_infections',
        location.id,
        start, end,
        label='Daily Infections',
        vlines=vlines,
    )
    ax_daily_infec.plot(
        full_data['date'],
        full_data['cumulative_cases'] - full_data['cumulative_cases'].shift(1),
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )

    make_time_plot(
        ax_daily_death,
        plot_versions,
        'daily_deaths',
        location.id,
        start, end,
        label='Daily Deaths',
        vlines=vlines,
    )
    ax_daily_death.plot(
        full_data['date'],
        full_data['cumulative_deaths'] - full_data['cumulative_deaths'].shift(1),
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )

    make_time_plot(
        ax_cumul_infec,
        plot_versions,
        'cumulative_infections',
        location.id,
        start, end,
        label='Cumulative Infected (% population)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop
    )

    make_time_plot(
        ax_cumul_death,
        plot_versions,
        'cumulative_deaths',
        location.id,
        start, end,
        label='Cumulative Deaths',
        vlines=vlines,
    )

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_time_plot(ax, plot_versions: List[PlotVersion], measure: str, loc_id: int,
                   start: pd.Timestamp, end: pd.Timestamp, label: str, vlines=(), transform=lambda x: x):
    for plot_version in plot_versions:
        data = plot_version.load_output_summaries(measure, loc_id)
        data[['mean', 'upper', 'lower']] = transform(data[['mean', 'upper', 'lower']])

        ax.plot(data['date'], data['mean'], color=plot_version.color)
        ax.fill_between(data['date'], data['upper'], data['lower'], alpha=FILL_ALPHA, color=plot_version.color)

    for vline_x in vlines:
        add_vline(ax, vline_x)

    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(DATE_LOCATOR)
    ax.xaxis.set_major_formatter(DATE_FORMATTER)

    ax.set_ylabel(label, fontsize=AX_LABEL_FONTSIZE)


def make_log_beta_resid_hist(ax, plot_versions: List[PlotVersion], loc_id: int):
    for plot_version in plot_versions:
        data = plot_version.load_output_draws('beta_scaling_parameters', loc_id)
        data = data[data.scaling_parameter == 'log_beta_residual_mean'].drop(columns='scaling_parameter').T

        ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='step')
        ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='stepfilled', alpha=FILL_ALPHA)

    ax.set_ylabel('count of log beta residual mean', fontsize=AX_LABEL_FONTSIZE)


def make_coefficient_plot(ax, plot_versions: List[PlotVersion], covariate: str, loc_id: int, label: str):
    for i, plot_version in enumerate(plot_versions):
        data = plot_version.load_output_draws('coefficients', loc_id)
        data = data[data.covariate == covariate].drop(columns='covariate').T
        # Boxplots are super annoying.
        props = dict(color=plot_version.color, linewidth=2)
        ax.boxplot(
            data,
            positions=[i],
            widths=[.7],
            boxprops=props,
            capprops=props,
            whiskerprops=props,
            flierprops={**props, **dict(markeredgecolor=plot_version.color)},
            medianprops=props,
            labels=[' '],  # Keeps the plot vertical size consistent with other things on the same row.
        )

    ax.set_ylabel(label, fontsize=AX_LABEL_FONTSIZE)


def add_vline(ax, x_position):
    ax.vlines(x_position, 0, 1, transform=ax.get_xaxis_transform(), linestyle='dashed', color='grey', alpha=0.8)


def make_title_and_legend(fig, location: Location, plot_versions: List[PlotVersion]):
    fig.suptitle(f'{location.name} ({location.id})', x=0.5, fontsize=TITLE_FONTSIZE, ha='left')
    fig.legend(handles=make_legend_handles(plot_versions),
               loc='lower center',
               bbox_to_anchor=(0.5, 0),
               fontsize=AX_LABEL_FONTSIZE,
               frameon=False,
               ncol=len(plot_versions))


def write_or_show(fig, plot_file: str):
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_legend_handles(plot_versions: List[PlotVersion]):
    handles = [mlines.Line2D([], [], color=pv.color, label=pv.label) for pv in plot_versions]
    return handles

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

COLOR_MAP = ['#7F3C8D', '#11A579',
             '#3969AC', '#F2B701',
             '#E73F74', '#80BA5A',
             '#E68310', '#008695',
             '#CF1C90', '#f97b72',
             '#4b4b8f', '#A5AA99'].__getitem__

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
    sns.set_style('whitegrid')
    # FIXME: This is brittle w/r/t new covariates
    time_varying = [c for c, c_config in COVARIATES.items() if c_config.time_varying]
    non_time_varying = [c for c, c_config in COVARIATES.items() if not c_config.time_varying]

        # Load some shared data.
    pv = plot_versions[0]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    full_data = pv.load_output_miscellaneous('full_data', is_table=True, location_id=location.id)
    #if not full_data.empty:
    #    data_date = full_data.loc[full_data.cumulative_deaths.notnull(), 'date'].max()
    #    short_term_forecast_date = data_date + SHORT_RANGE_FORECAST
    #    vlines = [data_date, short_term_forecast_date]
    #else:
    vlines = []

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=1,
                                 ncols=4,
                                 width_ratios=[1, 3, 5, 5],
                                 wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)

    sub_grid_wspace = 0.3
    gs_coef = grid_spec[0, 0].subgridspec(len(time_varying), 1)
    gs_cov = grid_spec[0, 1].subgridspec(len(time_varying), 1)
    gs_r = grid_spec[0, 2].subgridspec(2, 1)
    gs_beta = grid_spec[0, 3].subgridspec(3, 1, height_ratios=[2, 1, 1])

    ylim_map = {
        'mobility': (-100, 20),
        'testing': (0, 0.015),
        'pneumonia': (0.2, 1.5),
        'mask_use': (0, 1)
    }

    for i, covariate in enumerate(time_varying):
        ax_coef = fig.add_subplot(gs_coef[i])
        make_coefficient_plot(
            ax_coef,
            plot_versions,
            covariate,
            location.id,
            label=covariate.title(),
        )

        ax_cov = fig.add_subplot(gs_cov[i])
        make_time_plot(
            ax_cov,
            plot_versions,
            covariate,
            location.id,
            start, end,
            label=covariate.title(),
        )
        ylims = ylim_map.get(covariate)
        if ylims is not None:
            ax_cov.set_ylim(*ylims)

    ax_r_controlled = fig.add_subplot(gs_r[0])
    make_time_plot(
        ax_r_controlled,
        plot_versions,
        'r_controlled',
        location.id,
        start, end,
        vlines=vlines,
        label='R Controlled',
    )
    ax_r_controlled.set_ylim(0, 3)

    ax_r_eff = fig.add_subplot(gs_r[1])
    make_time_plot(
        ax_r_eff,
        plot_versions,
        'r_effective',
        location.id,
        start, end,
        label='R Effective',
        vlines=vlines,
    )
    ax_r_eff.set_ylim(0, 3)

    ax_betas = fig.add_subplot(gs_beta[0])
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

    ax_resid = fig.add_subplot(gs_beta[1])
    make_time_plot(
        ax_resid,
        plot_versions,
        'log_beta_residuals',
        location.id,
        start, end,
        label='Log Beta Residuals',
        vlines=vlines,
    )

    ax_rhist = fig.add_subplot(gs_beta[2])
    make_log_beta_resid_hist(
        ax_rhist,
        plot_versions,
        location.id,
    )

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_results_page(plot_versions: List[PlotVersion],
                      location: Location,
                      start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    sns.set_style('whitegrid')
    observed_color = COLOR_MAP(len(plot_versions))

    # Load some shared data.
    pv = plot_versions[0]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    full_data = pv.load_output_miscellaneous('full_data', is_table=True, location_id=location.id)
    #if not full_data.empty:
    #    data_date = full_data.loc[full_data.cumulative_deaths.notnull(), 'date'].max()
    #    short_term_forecast_date = data_date + SHORT_RANGE_FORECAST
    #    vlines = [data_date, short_term_forecast_date]
    #else:
    vlines = []

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=3, ncols=3, wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)


    ax_daily_infec = fig.add_subplot(grid_spec[0, 0])
    ax_daily_hosp = fig.add_subplot(grid_spec[1, 0])
    ax_daily_death = fig.add_subplot(grid_spec[2, 0])

    ax_cumul_infec = fig.add_subplot(grid_spec[0, 1])
    ax_census_hosp = fig.add_subplot(grid_spec[1, 1])
    ax_cumul_death = fig.add_subplot(grid_spec[2, 1])

    ax_susceptible = fig.add_subplot(grid_spec[1, 2])
    ax_immune = fig.add_subplot(grid_spec[0, 2])
    ax_es = fig.add_subplot(grid_spec[2, 2])

    # Column 1, Daily

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
        full_data['cumulative_cases'].diff(),
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )

    make_time_plot(
        ax_daily_hosp,
        plot_versions,
        'hospital_admissions',
        location.id,
        start, end,
        label='Daily Hospital and ICU Admissions',
        vlines=vlines,
        uncertainty=False,
    )
    make_time_plot(
        ax_daily_hosp,
        plot_versions,
        'icu_admissions',
        location.id,
        start, end,
        vlines=vlines,
        uncertainty=False,
        linestyle='dashed',
    )
    ax_daily_hosp.plot(
        full_data['date'],
        full_data['cumulative_hospitalizations'].diff(),
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
        full_data['cumulative_deaths'].diff(),
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )
    ax_daily_death.hlines([8 * pop / 1e6], start, end)
    # Column 2, Cumulative

    make_time_plot(
        ax_cumul_infec,
        plot_versions,
        'cumulative_infections',
        location.id,
        start, end,
        label='Cumulative Infected (% Population)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_cumul_infec.set_ylim(0, 100)

    make_time_plot(
        ax_census_hosp,
        plot_versions,
        'hospital_census',
        location.id,
        start, end,
        label='Hospital and ICU Census',
        vlines=vlines,
        uncertainty=False,
    )
    make_time_plot(
        ax_census_hosp,
        plot_versions,
        'icu_census',
        location.id,
        start, end,
        vlines=vlines,
        uncertainty=False,
        linestyle='dashed',
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

    # Column 3, miscellaneous

    make_time_plot(
        ax_susceptible,
        plot_versions,
        'total_susceptible',
        location.id,
        start, end,
        label='Total Susceptible (% Population)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_susceptible.set_ylim(0, 100)

    make_time_plot(
        ax_immune,
        plot_versions,
        'total_immune',
        location.id,
        start, end,
        label='Total Immune (% Population)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_immune.set_ylim(0, 100)
#    make_time_plot(
#        ax_immune,
#        plot_versions,
#        'herd_immunity',
#        location.id,
#        start, end,
#        transform=lambda x: 100*x,
#        uncertainty=False,
#        linestyle='dashed',
#    )
    # Todo: Add herd immunity to this plot

    # This is an overestimate, but not an important one.
#    es_end_date = pd.Timestamp.today() + SHORT_RANGE_FORECAST
#    make_time_plot(
#        ax_es,
#        plot_versions,
#        'daily_elastispliner_smoothed',
#        location.id,
#        start, es_end_date,
#        label='Elastispliner',
#    )
#    ax_es.plot(
#        full_data['date'],
#        full_data['cumulative_deaths'].diff(),
#        color=observed_color,
#        alpha=OBSERVED_ALPHA,
#    )
#    ax_es.scatter(
#        full_data['date'],
#        full_data['cumulative_deaths'].diff(),
#        color=observed_color,
#        alpha=OBSERVED_ALPHA,
#    )

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_time_plot(ax,
                   plot_versions: List[PlotVersion],
                   measure: str,
                   loc_id: int,
                   start: pd.Timestamp = None, end: pd.Timestamp = None,
                   label: str = None,
                   linestyle: str = 'solid',
                   uncertainty: bool = False,
                   vlines=(),
                   transform=lambda x: x):
    for plot_version in plot_versions:
        data = plot_version.load_output_summaries(measure, loc_id)
        data[['mean', 'upper', 'lower']] = transform(data[['mean', 'upper', 'lower']])

        ax.plot(data['date'], data['mean'], color=plot_version.color, linestyle=linestyle, linewidth=2.5)
        if uncertainty:
            ax.fill_between(data['date'], data['upper'], data['lower'], alpha=FILL_ALPHA, color=plot_version.color)

    for vline_x in vlines:
        add_vline(ax, vline_x)

    if start is not None and end is not None:
        date_locator = mdates.AutoDateLocator(maxticks=15)
        date_formatter = mdates.ConciseDateFormatter(date_locator, show_offset=False)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
    if label is not None:
        ax.set_ylabel(label, fontsize=AX_LABEL_FONTSIZE)
    sns.despine(ax=ax, left=True, bottom=True)


def make_log_beta_resid_hist(ax, plot_versions: List[PlotVersion], loc_id: int):
    for plot_version in plot_versions:
        data = plot_version.load_output_draws('beta_scaling_parameters', loc_id)
        data = data[data.scaling_parameter == 'log_beta_residual_mean'].drop(columns='scaling_parameter').T

        ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='step')
        ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='stepfilled', alpha=FILL_ALPHA)
    ax.set_xlim(-1, 1)
    ax.set_ylabel('count of log beta residual mean', fontsize=AX_LABEL_FONTSIZE)
    sns.despine(ax=ax, left=True, bottom=True)


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
    sns.despine(ax=ax, left=True, bottom=True)


def add_vline(ax, x_position):
    ax.vlines(x_position, 0, 1, transform=ax.get_xaxis_transform(), linestyle='dashed', color='grey', alpha=0.8)


def make_title_and_legend(fig, location: Location, plot_versions: List[PlotVersion]):
    fig.suptitle(f'{location.name} ({location.id})', x=0.5, fontsize=TITLE_FONTSIZE, ha='center')
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

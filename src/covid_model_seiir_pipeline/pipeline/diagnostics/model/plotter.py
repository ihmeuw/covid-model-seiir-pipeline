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
        make_variants_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_variants.pdf')
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
    vlines = []

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=1,
                                 ncols=4,
                                 width_ratios=[1, 3, 5, 5],
                                 wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_coef = grid_spec[0, 0].subgridspec(len(time_varying), 1)
    gs_cov = grid_spec[0, 1].subgridspec(len(time_varying), 1)
    gs_beta = grid_spec[0, 2].subgridspec(4, 1)
    gs_r = grid_spec[0, 3].subgridspec(4, 1)

    ylim_map = {
        'mobility': (-100, 20),
        'testing': (0, 0.015),
        'pneumonia': (0.2, 1.5),
        'mask_use': (0, 1),
        'variant_prevalence_B117': (0, 1),
        'variant_prevalence_B1351': (0, 1),
        'variant_prevalence_P1': (0, 1),
    }

    for i, covariate in enumerate(time_varying):
        ax_coef = fig.add_subplot(gs_coef[i])
        if 'variant' in covariate:
            label = covariate.split('_')[-1]
        else:
            label = covariate.title()
        make_coefficient_plot(
            ax_coef,
            plot_versions,
            covariate,
            location.id,
            label=label,
        )

        ax_cov = fig.add_subplot(gs_cov[i])
        make_time_plot(
            ax_cov,
            plot_versions,
            covariate,
            location.id,
            start, end,
            label=label,
        )
        ylims = ylim_map.get(covariate)
        if ylims is not None:
            ax_cov.set_ylim(*ylims)

    ax_beta_total = fig.add_subplot(gs_beta[0])
    make_time_plot(
        ax_beta_total,
        plot_versions,
        'betas',
        location.id,
        start, end,
        vlines=vlines,
        label='Regression Beta',
        transform=lambda x: np.log(x),
    )
    make_time_plot(
        ax_beta_total,
        plot_versions,
        'beta_hat',
        location.id,
        start, end,
        vlines=vlines,
        linestyle='dashed',
        transform=lambda x: np.log(x)
    )
    ax_beta_total.set_ylim(-3, 1)
    make_axis_legend(ax_beta_total, {'beta fit': {'linestyle': 'solid'},
                                     'beta hat': {'linestyle': 'dashed'}})

    ax_beta_wild = fig.add_subplot(gs_beta[1])
    make_time_plot(
        ax_beta_wild,
        plot_versions,
        'beta_wild',
        location.id,
        start, end,
        vlines=vlines,
        label='Non-escape Beta',
        transform=lambda x: np.log(x),
    )
    make_time_plot(
        ax_beta_wild,
        plot_versions,
        'empirical_beta_wild',
        location.id,
        start, end,
        vlines=vlines,
        linestyle='dashed',
        transform=lambda x: np.log(x)
    )
    ax_beta_wild.set_ylim(-3, 1)
    make_axis_legend(ax_beta_wild, {'input beta': {'linestyle': 'solid'},
                                    'empirical beta': {'linestyle': 'dashed'}})

    ax_beta_variant = fig.add_subplot(gs_beta[2])
    make_time_plot(
        ax_beta_variant,
        plot_versions,
        'beta_variant',
        location.id,
        start, end,
        vlines=vlines,
        label='Escape Beta',
        transform=lambda x: np.log(x),
    )
    make_time_plot(
        ax_beta_variant,
        plot_versions,
        'empirical_beta_variant',
        location.id,
        start, end,
        vlines=vlines,
        linestyle='dashed',
        transform=lambda x: np.log(x)
    )
    ax_beta_variant.set_ylim(-3, 1)
    make_axis_legend(ax_beta_variant, {'input beta': {'linestyle': 'solid'},
                                       'empirical': {'linestyle': 'dashed'}})

    ax_resid = fig.add_subplot(gs_beta[3])
    make_time_plot(
        ax_resid,
        plot_versions,
        'log_beta_residuals',
        location.id,
        start, end,
        label='Log Beta Residuals',
        vlines=vlines,
    )

    ax_reff = fig.add_subplot(gs_r[0])
    make_time_plot(
        ax_reff,
        plot_versions,
        'r_effective',
        location.id,
        start, end,
        label='R effective (system)',
        vlines=vlines,
    )
    ax_reff.set_ylim(-0.5, 3)

    ax_r_wild = fig.add_subplot(gs_r[1])
    make_placeholder(ax_r_wild, 'Non-escape R effective')

    ax_r_variant = fig.add_subplot(gs_r[2])
    make_placeholder(ax_r_variant, 'Escape R effective')

    ax_rhist = fig.add_subplot(gs_r[3])
    make_log_beta_resid_hist(
        ax_rhist,
        plot_versions,
        location.id,
    )

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_variants_page(plot_versions: List[PlotVersion],
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
    vlines = []

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=3, ncols=3, wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)

    ax_infecs_wild = fig.add_subplot(grid_spec[0, 0])
    ax_infecs_variant = fig.add_subplot(grid_spec[1, 0])
    ax_infecs = fig.add_subplot(grid_spec[2, 0])

    ax_infecs_nib = fig.add_subplot(grid_spec[0, 1])
    ax_infecs_vb = fig.add_subplot(grid_spec[1, 1])
    ax_variant_prev = fig.add_subplot(grid_spec[2, 1])

    ax_deaths_wild = fig.add_subplot(grid_spec[0, 2])
    ax_deaths_variant = fig.add_subplot(grid_spec[1, 2])
    ax_deaths = fig.add_subplot(grid_spec[2, 2])

    # Column 1

    make_time_plot(
        ax_infecs_wild,
        plot_versions,
        'daily_infections_wild',
        location.id,
        start, end,
        label='Daily Infections Wild',
        vlines=vlines,
    )
    make_time_plot(
        ax_infecs_variant,
        plot_versions,
        'daily_infections_variant',
        location.id,
        start, end,
        label='Daily Infections Variant',
        vlines=vlines,
    )
    make_time_plot(
        ax_infecs,
        plot_versions,
        'daily_infections',
        location.id,
        start, end,
        label='Daily Infections',
        vlines=vlines,
    )

    make_time_plot(
        ax_infecs_nib,
        plot_versions,
        'daily_infections_natural_immunity_breakthrough',
        location.id,
        start, end,
        label='Daily Infections Natural Breakthrough',
        vlines=vlines,
    )
    make_time_plot(
        ax_infecs_vb,
        plot_versions,
        'daily_infections_vaccine_breakthrough',
        location.id,
        start, end,
        label='Daily Infections Vaccine Breakthrough',
        vlines=vlines,
    )

    make_time_plot(
        ax_variant_prev,
        plot_versions,
        'escape_variant_prevalence',
        location.id,
        start, end,
        label='Escape Variant Prevalence',
        vlines=vlines,
    )
    make_time_plot(
        ax_variant_prev,
        plot_versions,
        'empirical_escape_variant_prevalence',
        location.id,
        start, end,
        vlines=vlines,
        linestyle='dashed',
    )
    ax_variant_prev.set_ylim([0, 1])
    make_axis_legend(ax_variant_prev, {'naive ramp': {'linestyle': 'solid'},
                                       'empirical': {'linestyle': 'dashed'}})

    make_time_plot(
        ax_deaths_wild,
        plot_versions,
        'daily_deaths_wild',
        location.id,
        start, end,
        label='Daily Deaths Wild',
        vlines=vlines,
    )
    make_time_plot(
        ax_deaths_variant,
        plot_versions,
        'daily_deaths_variant',
        location.id,
        start, end,
        label='Daily Deaths Variant',
        vlines=vlines,
    )
    make_time_plot(
        ax_deaths,
        plot_versions,
        'daily_deaths',
        location.id,
        start, end,
        label='Daily Deaths',
        vlines=vlines,
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
    pv = plot_versions[-1]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    full_data = pv.load_output_miscellaneous('full_data', is_table=True, location_id=location.id)
    vlines = []

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=4, ncols=4, wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)

    ax_daily_infec = fig.add_subplot(grid_spec[0, 0])
    ax_daily_case = fig.add_subplot(grid_spec[1, 0])
    ax_daily_hosp = fig.add_subplot(grid_spec[2, 0])
    ax_daily_death = fig.add_subplot(grid_spec[3, 0])

    ax_cumul_infec = fig.add_subplot(grid_spec[0, 1])
    ax_cumul_case = fig.add_subplot(grid_spec[1, 1])
    ax_census_hosp = fig.add_subplot(grid_spec[2, 1])
    ax_cumul_death = fig.add_subplot(grid_spec[3, 1])

    ax_vacc = fig.add_subplot(grid_spec[0, 2])
    ax_idr = fig.add_subplot(grid_spec[1, 2])
    ax_ihr = fig.add_subplot(grid_spec[2, 2])
    ax_ifr = fig.add_subplot(grid_spec[3, 2])

    ax_immune_wild = fig.add_subplot(grid_spec[0, 3])
    ax_immune_variant = fig.add_subplot(grid_spec[1, 3])
    ax_susceptible_wild = fig.add_subplot(grid_spec[2, 3])
    ax_susceptible_variant = fig.add_subplot(grid_spec[3, 3])

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
        ax_daily_case,
        plot_versions,
        'daily_cases',
        location.id,
        start, end,
        label='Daily Cases',
        vlines=vlines,
    )
    ax_daily_case.plot(
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
        label='Cumulative Infected (%)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_cumul_infec.set_ylim(0, 100)

    make_time_plot(
        ax_cumul_case,
        plot_versions,
        'cumulative_cases',
        location.id,
        start, end,
        label='Cumulative Cases',
        vlines=vlines,
    )

    make_time_plot(
        ax_census_hosp,
        plot_versions,
        'hospital_census',
        location.id,
        start, end,
        label='Hospital and ICU Census',
        vlines=vlines,
    )
    make_time_plot(
        ax_census_hosp,
        plot_versions,
        'icu_census',
        location.id,
        start, end,
        vlines=vlines,
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

    # Column 3, ratios
    make_time_plot(
        ax_vacc,
        plot_versions,
        'cumulative_vaccinations_effective',
        location.id,
        start, end,
        label='Cumulative Vaccinations (%)',
        vlines=vlines,
        transform=lambda x: x / pop * 100,
    )
    make_time_plot(
        ax_vacc,
        plot_versions,
        'cumulative_vaccinations_effective_input',
        location.id,
        start, end,
        vlines=vlines,
        linestyle='dashed',
    )

    idr = pv.load_output_summaries('infection_detection_ratio_es', location.id)
    make_time_plot(
        ax_idr,
        plot_versions,
        'infection_detection_ratio',
        location.id,
        start, end,
        label='IDR',
        vlines=vlines,
    )
    ax_idr.plot(
        idr['date'],
        idr['mean'],
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )

    ihr = pv.load_output_summaries('infection_hospitalization_ratio_es', location.id)
    make_time_plot(
        ax_ihr,
        plot_versions,
        'infection_hospitalization_ratio',
        location.id,
        start, end,
        label='IHR',
        vlines=vlines,
    )
    ax_ihr.plot(
        ihr['date'],
        ihr['mean'],
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )

    ifr = pv.load_output_summaries('infection_fatality_ratio_es', location.id)
    make_time_plot(
        ax_ifr,
        plot_versions,
        'infection_fatality_ratio',
        location.id,
        start, end,
        label='IFR',
        vlines=vlines,
    )
    ax_ifr.plot(
        ifr['date'],
        ifr['mean'],
        color=observed_color,
        alpha=OBSERVED_ALPHA,
    )

    # Column 4, miscellaneous
    make_time_plot(
        ax_immune_wild,
        plot_versions,
        'total_immune_wild',
        location.id,
        start, end,
        label='Immune Wild-type (%)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_immune_wild.set_ylim(0, 100)

    make_time_plot(
        ax_immune_variant,
        plot_versions,
        'total_immune_variant',
        location.id,
        start, end,
        label='Immune All Types (%)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_immune_variant.set_ylim(0, 100)

    make_time_plot(
        ax_susceptible_wild,
        plot_versions,
        'total_susceptible_wild',
        location.id,
        start, end,
        label='Susceptible Wild-type (%)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_susceptible_wild.set_ylim(0, 100)

    make_time_plot(
        ax_susceptible_variant,
        plot_versions,
        'total_susceptible_variant',
        location.id,
        start, end,
        label='Susceptible All Types (%)',
        vlines=vlines,
        transform=lambda x: 100 * x / pop,
    )
    ax_susceptible_variant.set_ylim(0, 100)

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
        try:
            data = plot_version.load_output_summaries(measure, loc_id)
        except FileNotFoundError:  # No data for this version, so skip.
            continue
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


def make_axis_legend(axis, elements: dict):
    handles = [mlines.Line2D([], [], label=e_name, linewidth=2.5, **e_props) for e_name, e_props in elements.items()]
    axis.legend(handles=handles,
                loc='upper left',
                fontsize=AX_LABEL_FONTSIZE,
                frameon=False)


def make_placeholder(axis, label: str):
    axis.axis([0, 1, 0, 1])
    axis.text(0.5, 0.5, f"TODO: {label}",
              verticalalignment='center', horizontalalignment='center',
              transform=axis.transAxes,
              fontsize=TITLE_FONTSIZE)


def write_or_show(fig, plot_file: str):
    if plot_file:
        fig.savefig(plot_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def make_legend_handles(plot_versions: List[PlotVersion]):
    handles = [mlines.Line2D([], [], color=pv.color, label=pv.label, linewidth=2.5) for pv in plot_versions]
    return handles

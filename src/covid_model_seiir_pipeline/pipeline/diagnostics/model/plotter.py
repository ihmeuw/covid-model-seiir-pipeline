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
    INFECTION_TO_CASE,
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
        make_results_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_results.pdf')
        )

        make_details_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_details.pdf')
        )

        make_drivers_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_drivers.pdf')
        )


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

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[5, 3, 5],
        wspace=0.2,
    )
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_daily = grid_spec[0, 0].subgridspec(3, 1)
    gs_rates = grid_spec[0, 1].subgridspec(6, 1)
    gs_infecs = grid_spec[0, 2].subgridspec(3, 1)

    plotter = Plotter(
        plot_versions=plot_versions,
        loc_id=location.id,
        start=start, end=end,
    )

    # Column 1, Daily
    daily_measures = [
        ('daily_cases', 'Daily Cases', 'cumulative_cases'),
        ('hospital_admissions', 'Daily Hospital Admissions', 'cumulative_hospitalizations'),
        ('daily_deaths', 'Daily Deaths', 'cumulative_deaths'),
    ]
    for i, (measure, label, full_data_measure) in enumerate(daily_measures):
        ax_measure = fig.add_subplot(gs_daily[i])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
        )
        ax_measure.scatter(
            full_data['date'],
            full_data[full_data_measure].diff(),
            color=observed_color,
            alpha=OBSERVED_ALPHA,
        )
        add_vline(ax_measure, full_data.loc[full_data[full_data_measure].notnull(), 'date'].max())

        if measure == 'daily_deaths':
            # Mandate reimposition level.
            ax_measure.hlines([8 * pop / 1e6], start, end)

    # Column 2, Cumulative & rates
    cumulative_measures = [
        ('daily_cases', 'Cumulative Cases'),
        ('hospital_admissions', 'Cumulative Hospital Admissions'),
        ('daily_deaths', 'Cumulative Deaths'),
    ]
    for i, (measure, label) in enumerate(cumulative_measures):
        ax_measure = fig.add_subplot(gs_rates[2*i])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
            transform=lambda x: x.cumsum(),
        )

    rates_measures = [
        ('infection_detection_ratio_es', 'infection_detection_ratio', 'IDR'),
        ('infection_hospitalization_ratio_es', 'infection_hospitalization_ratio', 'IHR'),
        ('infection_fatality_ratio_es', 'infection_fatality_ratio', 'IFR'),
    ]
    for i, (ies_measure, measure, label) in enumerate(rates_measures):
        rate = pv.load_output_summaries(ies_measure, location.id)
        ax_measure = fig.add_subplot(gs_rates[2*i + 1])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label
        )
        ax_measure.scatter(
            rate['date'],
            rate['mean'],
            color=observed_color,
            alpha=OBSERVED_ALPHA,
        )

    infections_measures = [
        ('daily_infections', 'Daily Infections'),
        ('cumulative_infections', 'Cumulative Infections (% Population)'),
        ('cumulative_infected', 'Cumulative Infected (% Population)'),
    ]
    for i, (measure, label) in enumerate(infections_measures):
        ax_measure = fig.add_subplot(gs_infecs[i])
        if 'cumulative' in measure:
            transform = lambda x: x / pop * 100
        else:
            transform = lambda x: x
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
            transform=transform,
        )
        if measure == 'daily_infections':
            ax_measure.scatter(
                full_data['date'],
                full_data['cumulative_cases'].diff().shift(-INFECTION_TO_CASE),
                color=observed_color,
                alpha=OBSERVED_ALPHA,
            )
            
    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_details_page(plot_versions: List[PlotVersion],
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
    hospital_census = pv.load_output_miscellaneous('hospital_census_data', is_table=True, location_id=location.id)

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(
        nrows=1, ncols=2,
        wspace=0.2,
    )
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_hospital = grid_spec[0, 0].subgridspec(3, 2)
    gs_detail = grid_spec[0, 1].subgridspec(2, 1, height_ratios=[3, 1])
    gs_infections = gs_detail[0, 0].subgridspec(3, 2)
    gs_susceptible = gs_detail[1, 0].subgridspec(1, 3)

    plotter = Plotter(
        plot_versions=plot_versions,
        loc_id=location.id,
        start=start, end=end,
    )

    # Hospital model section
    for i, measure in enumerate(['hospital', 'icu', 'ventilator']):
        if measure != 'ventilator':
            ax_daily = fig.add_subplot(gs_hospital[i, 0])
            plotter.make_time_plot(
                ax_daily,
                f'{measure}_admissions',
                label=f'{measure.title()} Admissions'
            )
            if measure == 'hospital':
                ax_daily.scatter(
                    full_data['date'],
                    full_data['cumulative_hospitalizations'].diff(),
                    color=observed_color,
                    alpha=OBSERVED_ALPHA,
                )
        ax_census = fig.add_subplot(gs_hospital[i, 1])
        plotter.make_time_plot(
            ax_census,
            f'{measure}_census',
            label=f'{measure.title()} Census',
        )
        ax_census.scatter(
            hospital_census['date'],
            hospital_census[f'{measure}_census'],
            color=observed_color,
            alpha=OBSERVED_ALPHA,
        )

    # Detailed infections section
    infections_measures = [
        ('daily_infections_wild', 'Daily Infections Non-Escape'),
        ('daily_infections_variant', 'Daily Infections Escape'),
        ('daily_infections_vaccine_breakthrough', 'Daily Infections Vaccine Escape'),
        ('daily_infections_natural_immunity_breakthrough', 'Daily Infections N. Immunity Escape'),
    ]
    for i, (measure, label) in enumerate(infections_measures):
        ax_measure = fig.add_subplot(gs_infections[i // 2, i % 2])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
        )

    susceptible_measures = [
        ('total_susceptible_wild', 'Naive Susceptible (% Population)'),
        ('total_susceptible_variant_only', 'Escape Variant Susceptible (% Population)'),
        ('total_susceptible_variant', 'Total Susceptible (% Population)'),
    ]
    for i, (measure, label) in enumerate(susceptible_measures):
        ax_measure = fig.add_subplot(gs_susceptible[i])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
            transform=lambda x: x / pop * 100,
        )
        ax_measure.set_ylim(0, 100)

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_drivers_page(plot_versions: List[PlotVersion],
                      location: Location,
                      start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    sns.set_style('whitegrid')

    # Load some shared data.
    pv = plot_versions[-1]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    time_varying = [c for c, c_config in COVARIATES.items() if c_config.time_varying]

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=2,
                                 ncols=4,
                                 width_ratios=[1, 4, 5, 5],
                                 height_ratios=[3, 1],
                                 wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_coef = grid_spec[0, 0].subgridspec(len(time_varying), 1)
    gs_cov = grid_spec[0, 1].subgridspec(len(time_varying), 1)
    gs_beta = grid_spec[:, 2].subgridspec(4, 1)
    gs_r = grid_spec[:, 3].subgridspec(4, 1)

    plotter = Plotter(
        plot_versions=plot_versions,
        loc_id=location.id,
        start=start, end=end,
    )

    ylim_map = {
        'mobility': (-100, 20),
        'testing': (0, 0.02),
        'pneumonia': (0.2, 1.5),
        'mask_use': (0, 1),
    }

    for i, covariate in enumerate(time_varying):
        ax_coef = fig.add_subplot(gs_coef[i])
        plotter.make_coefficient_plot(
            ax_coef,
            covariate,
            label=covariate.title(),
        )
        ax_cov = fig.add_subplot(gs_cov[i])
        plotter.make_time_plot(
            ax_cov,
            covariate,
            label=covariate.title(),
        )
        ylims = ylim_map.get(covariate)
        if ylims is not None:
            ax_cov.set_ylim(*ylims)

    ax_vaccine = fig.add_subplot(grid_spec[1, :2])

    plotter.make_time_plot(
        ax_vaccine,
        'cumulative_vaccinations_effective',
        label='Cumulative Vaccinations (%)',
        transform=lambda x: x / pop * 100,
    )
    plotter.make_time_plot(
        ax_vaccine,
        'cumulative_vaccinations_effective_input',
        linestyle='dashed',
        transform=lambda x: x / pop * 100,
    )
    make_axis_legend(ax_vaccine, {'effective delivered': {'linestyle': 'solid'},
                                  'effective available': {'linestyle': 'dashed'}})
    ax_vaccine.set_ylim(0, 100)

    beta_measures = [
        ('betas', 'beta_hat', 'Regression Beta (log)', ('beta final', 'beta hat')),
        ('beta_wild', 'empirical_beta_wild', 'Non-Escape Beta (log)', ('input beta', 'empirical beta')),
        ('beta_variant', 'empirical_beta_variant', 'Escape Beta (log)', ('input beta', 'empirical beta')),
    ]
    for i, (measure1, measure2, label, legend_names) in enumerate(beta_measures):
        ax_measure = fig.add_subplot(gs_beta[i])
        plotter.make_time_plot(
            ax_measure,
            measure1,
            label=label,
            transform=lambda x: np.log(x),
        )
        plotter.make_time_plot(
            ax_measure,
            measure2,
            linestyle='dashed',
            transform=lambda x: np.log(x),
        )
        ax_measure.set_ylim(-3, 1)
        make_axis_legend(
            ax_measure,
            {name: {'linestyle': style} for name, style in zip(legend_names, ['solid', 'dashed'])}
        )

    ax_rho = fig.add_subplot(gs_beta[-1])
    plotter.make_time_plot(
        ax_rho,
        'non_escape_variant_prevalence',
        label='Non-Escape Variant Prevalence',
    )

    ax_resid = fig.add_subplot(gs_r[0])
    plotter.make_time_plot(
        ax_resid,
        'log_beta_residuals',
        label='Log Beta Residuals',
    )
    ax_rhist = fig.add_subplot(gs_r[1])
    plotter.make_log_beta_resid_hist(
        ax_rhist,
    )

    ax_reff = fig.add_subplot(gs_r[2])
    plotter.make_time_plot(
        ax_reff,
        'r_effective',
        label='R effective (empirical)',
    )
    ax_reff.set_ylim(-0.5, 3)

    ax_rho_escape = fig.add_subplot(gs_r[-1])
    plotter.make_time_plot(
        ax_rho_escape,
        'escape_variant_prevalence',
        label='Escape Variant Prevalence',
    )
    plotter.make_time_plot(
        ax_rho_escape,
        'empirical_escape_variant_prevalence',
        linestyle='dashed',
    )
    ax_rho_escape.set_ylim([0, 1])
    make_axis_legend(ax_rho_escape, {'naive ramp': {'linestyle': 'solid'},
                                     'empirical': {'linestyle': 'dashed'}})

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


class Plotter:

    def __init__(self,
                 plot_versions: List[PlotVersion],
                 loc_id: int,
                 start: pd.Timestamp,
                 end: pd.Timestamp,
                 uncertainty: bool = False,
                 transform=lambda x: x,
                 **extra_defaults):
        self._plot_versions = plot_versions
        self._loc_id = loc_id
        self._start = start
        self._end = end

        self._uncertainty = uncertainty
        self._transform = transform
        self._default_options = {'linewidth': 2.5, **extra_defaults}

    def make_time_plot(self, ax, measure: str, label: str = None, **extra_options):
        uncertainty = extra_options.pop('uncertainty', self._uncertainty)
        transform = extra_options.pop('transform', self._transform)
        start = extra_options.pop('start', self._start)
        end = extra_options.pop('end', self._end)

        plot_options = {**self._default_options, **extra_options}

        for plot_version in self._plot_versions:
            try:
                data = plot_version.load_output_summaries(measure, self._loc_id)
            except FileNotFoundError:  # No data for this version, so skip.
                continue
            data[['mean', 'upper', 'lower']] = transform(data[['mean', 'upper', 'lower']])

            ax.plot(data['date'], data['mean'], color=plot_version.color, **plot_options)
            if uncertainty:
                ax.fill_between(data['date'], data['upper'], data['lower'], alpha=FILL_ALPHA, color=plot_version.color)

        date_locator = mdates.AutoDateLocator(maxticks=15)
        date_formatter = mdates.ConciseDateFormatter(date_locator, show_offset=False)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)

        if label is not None:
            ax.set_ylabel(label, fontsize=AX_LABEL_FONTSIZE)

    def make_log_beta_resid_hist(self, ax):
        for plot_version in self._plot_versions:
            data = plot_version.load_output_draws('beta_scaling_parameters', self._loc_id)
            data = data[data.scaling_parameter == 'log_beta_residual_mean'].drop(columns='scaling_parameter').T

            ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='step')
            ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='stepfilled', alpha=FILL_ALPHA)
        ax.set_xlim(-1, 1)
        ax.set_ylabel('count of log beta residual mean', fontsize=AX_LABEL_FONTSIZE)
        sns.despine(ax=ax, left=True, bottom=True)

    def make_coefficient_plot(self, ax, covariate: str, label: str):
        for i, plot_version in enumerate(self._plot_versions):
            data = plot_version.load_output_draws('coefficients', self._loc_id)
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
    if not pd.isnull(x_position):
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
    handles = [mlines.Line2D([], [], label=e_name, linewidth=2.5, color='k', **e_props)
               for e_name, e_props in elements.items()]
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

from collections import defaultdict
from pathlib import Path
from typing import List
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
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
OBSERVED_ALPHA = 0.3
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

        make_drivers_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_drivers.pdf')
        )

        make_variant_page(
            plot_versions,
            location,
            date_start, date_end,
            plot_file=str(output_dir / f'{location.id}_variant.pdf')
        )


def make_results_page(plot_versions: List[PlotVersion],
                      location: Location,
                      start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    sns.set_style('whitegrid')

    # Load some shared data.
    pv = plot_versions[-1]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    full_data = pv.load_output_miscellaneous('unscaled_full_data', is_table=True, location_id=location.id)
    full_data = full_data.set_index('date')
    em_scalars = pv.load_output_miscellaneous('excess_mortality_scalars', is_table=True, location_id=location.id)
    em_scalars = em_scalars.set_index('date').reindex(full_data.index).ffill().bfill()
    daily_deaths = full_data['cumulative_deaths'].diff().fillna(full_data['cumulative_deaths'])
    full_data['cumulative_deaths'] = (em_scalars['mean'] * daily_deaths).cumsum()
    full_data = full_data.reset_index()

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
    group_axes = []
    daily_measures = [
        ('daily_cases', 'Daily Cases', 'cumulative_cases'),
        ('daily_admissions', 'Daily Admissions', 'cumulative_hospitalizations'),
        ('daily_deaths', 'Daily Deaths', 'cumulative_deaths'),
    ]
    for i, (measure, label, full_data_measure) in enumerate(daily_measures):
        ax_measure = fig.add_subplot(gs_daily[i])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
        )
        plotter.make_observed_time_plot(
             ax_measure,
             full_data['date'],
             full_data[full_data_measure].diff().fillna(full_data[full_data_measure]),
        )

        # if measure == 'daily_deaths':
        #     # Mandate reimposition level.
        #     ax_measure.hlines([8 * pop / 1e6], start, end)
        group_axes.append(ax_measure)
    fig.align_ylabels(group_axes)

    # Column 2, Cumulative & rates
    group_axes = []
    cumulative_measures = [
        ('daily_cases', 'Cumulative Cases', 'cumulative_cases'),
        ('daily_admissions', 'Cumulative Admissions', 'cumulative_hospitalizations'),
        ('daily_deaths', 'Cumulative Deaths', 'cumulative_deaths'),
    ]
    for i, (measure, label, full_data_measure) in enumerate(cumulative_measures):
        ax_measure = fig.add_subplot(gs_rates[2*i])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
            transform=lambda x: x.cumsum(),
        )
        plotter.make_observed_time_plot(
            ax_measure,
            full_data['date'],
            full_data[full_data_measure],
        )
        group_axes.append(ax_measure)

    rates_measures = [
        ('infection_detection_ratio_es', 'infection_detection_ratio', 'IDR'),
        ('infection_hospitalization_ratio_es', 'infection_hospitalization_ratio', 'IHR'),
        ('infection_fatality_ratio_es', 'infection_fatality_ratio', 'IFR'),
    ]
    for i, (ies_measure, measure, label) in enumerate(rates_measures):
        ax_measure = fig.add_subplot(gs_rates[2*i + 1])
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label
        )
        group_axes.append(ax_measure)

        # # Sometimes weird stuff at the start of the series.
        # if len(rate):
        #     y_max = rate['mean'].dropna().iloc[100:].max()
        #     ax_measure.set_ylim(0, y_max)

    fig.align_ylabels(group_axes)

    group_axes = []
    infections_measures = [
        ('daily_infections', 'Daily Infections'),
        ('cumulative_infections', 'Cumulative Infections (%)'),
        ('effective_susceptible_total', 'Effectively Susceptible (%)'),
    ]
    for i, (measure, label) in enumerate(infections_measures):
        ax_measure = fig.add_subplot(gs_infecs[i])
        if 'cumulative' in measure or 'effective' in measure:
            transform = lambda x: x / pop * 100
        else:
            transform = lambda x: x
        plotter.make_time_plot(
            ax_measure,
            measure,
            label=label,
            transform=transform,
        )
        group_axes.append(ax_measure)
    fig.align_ylabels(group_axes)

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
    full_data_unscaled = pv.load_output_miscellaneous('unscaled_full_data', is_table=True,
                                                      location_id=location.id)
    hospital_census = pv.load_output_miscellaneous('hospital_census_data', is_table=True,
                                                   location_id=location.id)

    time_varying = [c for c, c_config in COVARIATES.items() if c_config.time_varying]

    # Configure the plot layout.
    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    grid_spec = fig.add_gridspec(nrows=5,
                                 ncols=3,
                                 width_ratios=[1, 4, 6],
                                 wspace=0.2)
    grid_spec.update(**GRID_SPEC_MARGINS)

    gs_coef = grid_spec[:, 0].subgridspec(len(time_varying) + 2, 1)
    gs_cov = grid_spec[:, 1].subgridspec(len(time_varying) + 2, 1)
    gs_hospital = grid_spec[0:3, 2].subgridspec(3, 2)
    gs_other = grid_spec[3:, 2].subgridspec(2, 1)
    gs_vax = gs_other[0].subgridspec(1, 2)
    gs_deaths = gs_other[1].subgridspec(1, 3, wspace=0.25)

    plotter = Plotter(
        plot_versions=plot_versions,
        loc_id=location.id,
        start=start, end=end,
    )

    ylim_map = {
        'mandates_index_1': (0.0, 1.0),
        'mandates_index_2': (0.0, 1.0),
        'mobility': (-100, 50),
        'testing': (0, 0.02),
        'pneumonia': (0.2, 1.5),
        'mask_use': (0, 1),
    }
    coef_axes, cov_axes = [], []
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
        coef_axes.append(ax_coef)
        cov_axes.append(ax_cov)
    ax_intercept = fig.add_subplot(gs_coef[-2])
    plotter.make_coefficient_plot(
        ax_intercept,
        'intercept',
        label='Intercept',
    )
    coef_axes.append(ax_intercept)

    ax_beta = fig.add_subplot(gs_cov[-2])
    plotter.make_time_plot(
        ax_beta,
        'betas',
        label='Log Beta',
        transform=lambda x: np.log(x),
    )
    plotter.make_time_plot(
        ax_beta,
        'beta_hat',
        linestyle='dashed',
        transform=lambda x: np.log(x),
    )
    ax_beta.set_ylim(-3, 1)
    make_axis_legend(
        ax_beta,
        {name: {'linestyle': style} for name, style in
         zip(('beta final', 'beta hat'), ['solid', 'dashed'])}
    )
    cov_axes.append(ax_beta)

    ax_hist = fig.add_subplot(gs_coef[-1])
    plotter.make_log_beta_resid_hist(
        ax_hist,
    )
    coef_axes.append(ax_hist)

    ax_resid = fig.add_subplot(gs_cov[-1])
    plotter.make_time_plot(
        ax_resid,
        'log_beta_residuals',
        label='Log Beta Residuals',
    )
    plotter.make_time_plot(
        ax_resid,
        'scaled_log_beta_residuals',
        linestyle='dashed',
    )
    cov_axes.append(ax_resid)

    fig.align_ylabels(coef_axes)
    fig.align_ylabels(cov_axes)

    # Hospital model section
    axes = defaultdict(list)
    for col, (measure, observed_measure) in enumerate(
            [('daily', 'hospital'), ('icu', 'icu')]):
        ax_daily = fig.add_subplot(gs_hospital[0, col])
        label = measure.upper() if measure == 'icu' else measure.title()
        plotter.make_time_plot(
            ax_daily,
            f'{measure}_admissions',
            label=f'{label} Admissions'
        )

        if observed_measure == 'hospital':
            plotter.make_observed_time_plot(
                ax_daily,
                full_data_unscaled['date'],
                full_data_unscaled['cumulative_hospitalizations'].diff(),
            )
        axes[col].append(ax_daily)

        ax_correction = fig.add_subplot(gs_hospital[1, col])
        plotter.make_time_plot(
            ax_correction,
            f'{observed_measure}_census_correction_factor',
            label=f'{label} Census Correction Factor'
        )
        axes[col].append(ax_correction)

        ax_census = fig.add_subplot(gs_hospital[2, col])
        plotter.make_time_plot(
            ax_census,
            f'{observed_measure}_census',
            label=f'{label} Census',
        )
        plotter.make_observed_time_plot(
            ax_census,
            hospital_census['date'],
            hospital_census[f'{observed_measure}_census'],
        )
        axes[col].append(ax_census)
    for ax_set in axes.values():
        fig.align_ylabels(ax_set)

    for i, label in enumerate(['vaccinations', 'boosters']):
        ax_vaccine = fig.add_subplot(gs_vax[i])

        plotter.make_time_plot(
            ax_vaccine,
            f'daily_{label}',
            label=f'Cumulative {label.capitalize()} (%)',
            transform=lambda x: x / pop * 100,
        )
        ax_vaccine.set_ylim(0, 100)

    ax_unscaled = fig.add_subplot(gs_deaths[0])
    plotter.make_time_plot(
        ax_unscaled,
        'unscaled_daily_deaths',
        label='Unscaled Deaths',
    )
    plotter.make_observed_time_plot(
        ax_unscaled,
        full_data_unscaled['date'],
        full_data_unscaled['cumulative_deaths'].diff(),
    )

    ax_scalars = fig.add_subplot(gs_deaths[1])
    for plot_version in plot_versions:
        try:
            data = plot_version.load_output_miscellaneous(
                'excess_mortality_scalars',
                is_table=True,
                location_id=location.id
            )

            ax_scalars.plot(data['date'], data['mean'], color=plot_version.color)
            if plotter._uncertainty:
                ax_scalars.fill_between(data['date'], data['upper'], data['lower'],
                                        alpha=FILL_ALPHA, color=plot_version.color)
        except (KeyError, FileNotFoundError):
            pass

    ax_scalars.set_ylabel('Total COVID Scalar', fontsize=AX_LABEL_FONTSIZE)
    plotter.format_date_axis(ax_scalars)

    ax_scaled = fig.add_subplot(gs_deaths[2])
    plotter.make_time_plot(
        ax_scaled,
        'daily_deaths',
        label='Deaths',
    )

    ax_unscaled.set_ylim(ax_scaled.get_ylim())

    make_title_and_legend(fig, location, plot_versions)
    write_or_show(fig, plot_file)


def make_variant_page(plot_versions: List[PlotVersion],
                      location: Location,
                      start: pd.Timestamp, end: pd.Timestamp,
                      plot_file: str = None):
    sns.set_style('whitegrid')

    # Load some shared data.
    pv = plot_versions[-1]
    pop = pv.load_output_miscellaneous('populations', is_table=True, location_id=location.id)
    pop = pop.loc[(pop.age_group_id == 22) & (pop.sex_id == 3), 'population'].iloc[0]

    variant_prevalence = pv.load_output_miscellaneous('variant_prevalence', is_table=True,
                                                      location_id=location.id).set_index('date')

    # Configure the plot layout.
    fig = plt.figure(figsize=(40, 20), tight_layout=True)

    measures = [
        'daily_infections',
        'effective_susceptible',
        # 'effective_immune',
        'force_of_infection',
        'r_effective',
        'r_controlled',
        'variant_prevalence',
    ]

    grid_spec = fig.add_gridspec(
        nrows=len(measures), ncols=len(VARIANT_NAMES) - 2,
        wspace=0.2,
    )
    grid_spec.update(**GRID_SPEC_MARGINS)

    plotter = Plotter(
        plot_versions=plot_versions,
        loc_id=location.id,
        start=start, end=end,
    )

    axes = defaultdict(list)

    for row, measure in enumerate(measures):
        for col, variant in enumerate(['ancestral', 'alpha', 'beta', 'gamma', 'delta', 'omicron', 'ba5', 'omega']):
            ax = fig.add_subplot(grid_spec[row, col])
            key = f'{measure}_{variant}'

            if measure in ['effective_susceptible', 'effective_immune']:
                transform = lambda x: x / pop * 100
            else:
                transform = lambda x: x

            plotter.make_time_plot(
                ax,
                key,
                '',
                transform=transform,
            )
            if measure == 'effective_susceptible':
                if variant in ['ancestral', 'alpha']:
                    key = 'total_susceptible_wild'
                else:
                    key = 'total_susceptible_variant'
                plotter.make_time_plot(
                    ax,
                    key,
                    label='',
                    transform=lambda x: x / pop * 100,
                )

            if measure == 'daily_infections':
                if variant in ['ancestral', 'alpha']:
                    key = 'total_susceptible_wild'
                else:
                    key = 'total_susceptible_variant'

            if measure == 'variant_prevalence':
                observed_color = COLOR_MAP(len(plot_versions))
                variant_prevalence.loc[:, f'{variant}'].plot(ax=ax, linewidth=3, color=observed_color,
                                                             linestyle='dashed')

            if measure in ['r_effective', 'r_controlled']:
                ax.set_ylim(0, 4)
            elif measure in ['effective_susceptible', 'effective_immune']:
                ax.set_ylim(0, 100)
            elif measure == 'variant_prevalence':
                ax.set_ylim(0, 1)

            if row == 0:
                ax.set_title(variant.title(), fontsize=18)

            if col == 0:
                ax.set_ylabel(measure.replace('_', ' ').title(), fontsize=18)
            axes[col].append(ax)

    for col, ax_group in axes.items():
        fig.align_ylabels(ax_group)

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
                data = data[(start <= data['date']) & (data['date'] <= end)]
            except FileNotFoundError:  # No data for this version, so skip.
                continue
            data[['mean', 'upper', 'lower']] = transform(data[['mean', 'upper', 'lower']])

            ax.plot(data['date'], data['mean'], color=plot_version.color, **plot_options)
            if uncertainty:
                ax.fill_between(data['date'], data['upper'], data['lower'], alpha=FILL_ALPHA, color=plot_version.color)

        self.format_date_axis(ax, start, end)

        if label is not None:
            ax.set_ylabel(label, fontsize=AX_LABEL_FONTSIZE)

    def make_observed_time_plot(self, ax, x, y):
        observed_color = COLOR_MAP(len(self._plot_versions))
        ax.scatter(
            x, y,
            color=observed_color,
            alpha=OBSERVED_ALPHA,
        )
        ax.plot(
            x, y,
            color=observed_color,
            alpha=OBSERVED_ALPHA,
        )

    def make_raw_time_plot(self, ax, data, measure, label):
        ax.plot(
            data['date'],
            data[measure],
            color=self._plot_versions[-1].color,
            linewidth=self._default_options['linewidth'],
        )
        ax.set_ylabel(label, fontsize=AX_LABEL_FONTSIZE)
        self.format_date_axis(ax)

    def make_log_beta_resid_hist(self, ax):
        for plot_version in self._plot_versions:
            data = plot_version.load_output_draws('beta_scaling_parameters', self._loc_id)
            data = data[data.scaling_parameter == 'log_beta_residual_mean'].drop(columns='scaling_parameter').T

            ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='step')
            ax.hist(data.values, color=plot_version.color, bins=HIST_BINS, histtype='stepfilled', alpha=FILL_ALPHA)
        ax.set_xlim(-1, 1)
        ax.set_ylabel('Residual count', fontsize=AX_LABEL_FONTSIZE)
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

    def format_date_axis(self, ax, start=None, end=None):
        start = start if start is not None else self._start
        end = end if end is not None else self._end
        date_locator = mdates.AutoDateLocator(maxticks=15)
        date_formatter = mdates.ConciseDateFormatter(date_locator, show_offset=False)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)


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

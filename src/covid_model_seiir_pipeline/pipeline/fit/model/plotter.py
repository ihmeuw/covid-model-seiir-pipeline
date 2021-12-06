from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)

from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface

logger = cli_tools.task_performance_logger


_COLOR_MEASURES = (
    ({'light': 'indianred', 'dark': 'darkred'}, ['death', 'deaths', 'ifr']),
    ({'light': 'mediumseagreen', 'dark': 'darkgreen'}, ['case', 'cases', 'idr']),
    ({'light': 'dodgerblue', 'dark': 'navy'}, ['admission', 'admissions', 'hospitalizations', 'ihr']),
)
MEASURE_COLORS = {
    measure: cmap for cmap, measures in _COLOR_MEASURES for measure in measures
}


class Location(NamedTuple):
    id: int
    name: str


def ies_plot(location: Location,
             data_dictionary: Dict[str, pd.DataFrame],
             data_interface: FitDataInterface,
             start: pd.Timestamp = pd.Timestamp('2020-02-01'),
             end: pd.Timestamp = pd.Timestamp.today(),
             uncertainty: bool = False,
             review: bool = False,
             plot_root: str = None):
    pop = data_interface.load_population('total').population.loc[location.id]
    plotter = Plotter(
        data_dictionary=data_dictionary,
        location=location,
        start=start,
        end=end,
        uncertainty=uncertainty,
    )

    # Configure the plot layout.
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=plotter.fig_size, tight_layout=True)
    grid_spec = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[5, 3, 5],
        wspace=0.2,
    )
    grid_spec.update(**plotter.grid_spec_margins)

    gs_daily = grid_spec[0, 0].subgridspec(3, 1)
    gs_rates = grid_spec[0, 1].subgridspec(6, 1)
    gs_infecs = grid_spec[0, 2].subgridspec(3, 1)

    # Column 1, Daily
    group_axes = []
    daily_measures = [
        ('cases', 'Daily Cases'),
        ('hospitalizations', 'Daily Admissions'),
        ('deaths', 'Daily Deaths'),
    ]
    for i, (measure, label) in enumerate(daily_measures):
        ax_measure = fig.add_subplot(gs_daily[i])

        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'smoothed_daily_{measure}',
            color=MEASURE_COLORS[measure]['dark'],
            label=label
        )
        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'posterior_daily_{measure}',
            color=MEASURE_COLORS[measure]['light'],
            linestyle='--',
        )
        plotter.make_observed_time_plot(
            ax=ax_measure,
            measure=f'daily_{measure}',
            color_dict=MEASURE_COLORS[measure],
        )

        group_axes.append(ax_measure)

    plotter.clean_and_align_axes(fig, group_axes)

    # Column 2, Cumulative & rates
    group_axes = []
    cumulative_measures = [
        ('cases', 'Cumulative Cases'),
        ('hospitalizations', 'Cumulative Admissions'),
        ('deaths', 'Cumulative Deaths'),
    ]
    for i, (measure, label) in enumerate(cumulative_measures):
        ax_measure = fig.add_subplot(gs_rates[ 2 *i])

        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'smoothed_cumulative_{measure}',
            color=MEASURE_COLORS[measure]['dark'],
            label=label,
        )
        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'posterior_cumulative_{measure}',
            color=MEASURE_COLORS[measure]['light'],
            linestyle='--',
        )

        plotter.make_observed_time_plot(
            ax=ax_measure,
            measure=f'cumulative_{measure}',
            color_dict=MEASURE_COLORS[measure],
        )

        group_axes.append(ax_measure)

    rates_measures = [
        ('idr', (0, 100)),
        ('ihr', (0, 10)),
        ('ifr', (0, 5)),
    ]
    for i, (measure, ylim) in enumerate(rates_measures):
        ax_measure = fig.add_subplot(gs_rates[ 2 *i + 1])

        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'prior_{measure}',
            transform=lambda x: x * 100,
            color=MEASURE_COLORS[measure]['dark'],
            label=f'{measure.upper()} (%)',
        )
        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'posterior_{measure}',
            transform=lambda x: x * 100,
            color=MEASURE_COLORS[measure]['light'],
            linestyle='--',
        )
        ax_measure.set_ylim(ylim)

        try:
            rates_data = data_dictionary['rates_data'].loc[(measure, location.id)]
            ax_measure.scatter(
                rates_data.index,
                rates_data.values * 100,
                color=MEASURE_COLORS[measure]['dark'],
                marker='o',
                facecolors='none'
            )
        except KeyError:
            pass

        group_axes.append(ax_measure)

    plotter.clean_and_align_axes(fig, group_axes)

    ax_daily = fig.add_subplot(gs_infecs[0])
    ax_cumul = fig.add_subplot(gs_infecs[1])

    for metric, ax, transform in [('daily', ax_daily, identity), ('cumulative', ax_cumul, pop_scale(pop))]:
        for measure in ['cases', 'deaths', 'hospitalizations']:
            plotter.make_time_plot(
                ax=ax,
                measure=f'posterior_{measure}_based_{metric}_naive_unvaccinated_infections',
                color=MEASURE_COLORS[measure]['light'],
                linestyle='--',
                transform=transform,
            )

        suffix = ' (%)' if metric == 'cumulative' else ''
        plotter.make_time_plot(
            ax=ax,
            measure=f'posterior_{metric}_naive_unvaccinated_infections',
            color='black',
            label=f'{metric.capitalize()} Infections{suffix}',
            uncertainty=True,
            transform=transform,
        )

    try:
        sero_data = data_dictionary['seroprevalence'].loc[location.id]
    except KeyError:
        sero_data = None
    if sero_data is not None:
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 1].index,
                         sero_data.loc[sero_data['is_outlier'] == 1, 'reported_seroprevalence'] * 100,
                         s=80, c='maroon', alpha=0.45, marker='x')
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 0].index,
                         sero_data.loc[sero_data['is_outlier'] == 0, 'reported_seroprevalence'] * 100,
                         s=80, c='darkturquoise', edgecolors='darkcyan', alpha=0.3, marker='s')
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 0].index,
                         sero_data.loc[sero_data['is_outlier'] == 0, 'seroprevalence'] * 100,
                         s=80, c='orange', edgecolors='darkorange', alpha=0.3, marker='^')
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 0].index,
                         sero_data.loc[sero_data['is_outlier'] == 0, 'adjusted_seroprevalence'] * 100,
                         s=100, c='mediumorchid', edgecolors='darkmagenta', alpha=0.6, marker='o')

    group_axes = [ax_daily, ax_cumul]

    if review:
        ax_stackplot = fig.add_subplot(gs_infecs[2])
        naive_unvax = data_dictionary['posterior_cumulative_naive_unvaccinated_infections'].loc[location.id, 'mean'] * 100 / pop
        naive = data_dictionary['posterior_cumulative_naive_infections'].loc[location.id, 'mean'] * 100 / pop
        total = data_dictionary['posterior_cumulative_total_infections'].loc[location.id, 'mean'] * 100 / pop
        ax_stackplot.stackplot(
            naive_unvax.index,
            naive_unvax,
            naive - naive_unvax,
            total - naive,
            )
        ax_stackplot.set_ylabel('Cumulative Infections (%)', fontsize=plotter.ax_label_fontsize)
        plotter.format_date_axis(ax_stackplot)

        group_axes.append(ax_stackplot)
    else:
        ax_beta = fig.add_subplot(gs_infecs[2])
        for measure in ['deaths', 'hospitalizations', 'cases']:
            plotter.make_time_plot(
                ax=ax_beta,
                measure=f'beta_{measure}',
                color=MEASURE_COLORS[measure]['light'],
                linestyle='--',
                transform=np.log,
            )
        plotter.make_time_plot(
            ax=ax_beta,
            measure='beta',
            color='black',
            label='Log Beta',
            transform=np.log,
            uncertainty=True,
        )
        ax_beta.set_ylim(-2, 3)

        group_axes.append(ax_beta)

    plotter.clean_and_align_axes(fig, group_axes)

    sns.despine(fig=fig, left=True, bottom=True)
    fig.suptitle(f'{location.name} ({location.id})', x=0.5, fontsize=plotter.title_fontsize, ha='center')

    if plot_root:
        plot_path = Path(plot_root) / f'ies_{location.id}.pdf'
    else:
        plot_path = None
    write_or_show(fig, plot_path)


def identity(x):
    return x


def pop_scale(pop):
    def _scale(x):
        return 100 * x / pop

    return _scale


class Plotter:
    fill_alpha = 0.15
    observed_alpha = 0.3
    ax_label_fontsize = 16
    tick_label_fontsize = 11
    title_fontsize = 24
    fig_size = (30, 15)
    grid_spec_margins = {'top': 0.92, 'bottom': 0.08}

    def __init__(self,
                 data_dictionary: Dict[str, pd.DataFrame],
                 location: Location,
                 start: pd.Timestamp,
                 end: pd.Timestamp,
                 uncertainty: bool,
                 transform=identity,
                 **extra_defaults):
        self._data_dictionary = {}
        for measure, data in data_dictionary.items():
            try:
                self._data_dictionary[measure] = data.loc[location.id]
            except KeyError:
                self._data_dictionary[measure] = data

        self._location = location
        self._start = start
        self._end = end

        self._uncertainty = uncertainty
        self._transform = transform
        self._default_options = {'linewidth': 2.5, **extra_defaults}

    def make_time_plot(self, ax, measure: str, color: str, label: str = None, **extra_options):
        uncertainty = extra_options.pop('uncertainty', self._uncertainty)
        transform = extra_options.pop('transform', self._transform)
        start = extra_options.pop('start', self._start)
        end = extra_options.pop('end', self._end)

        plot_options = {**self._default_options, **extra_options}

        data = self._data_dictionary[measure].reset_index()
        data = data[(start <= data['date']) & (data['date'] <= end)]

        transform_cols = [c for c in data if c in ['mean', 'upper', 'lower', 'upper2', 'lower2']]
        data[transform_cols] = transform(data[transform_cols])

        ax.plot(data['date'], data['mean'], color=color, **plot_options)
        if uncertainty:
            ax.fill_between(data['date'], data['upper'], data['lower'], alpha=self.fill_alpha, color=color)
            if 'upper2' in data:
                ax.fill_between(data['date'], data['upper2'], data['lower2'], alpha=1.5 * self.fill_alpha, color=color)

        self.format_date_axis(ax, start, end)

        if label is not None:
            ax.set_ylabel(label, fontsize=self.ax_label_fontsize)
            ax.tick_params(axis='y', labelsize=self.tick_label_fontsize)

    def make_observed_time_plot(self, ax, measure: str, color_dict: Dict[str, str], **extra_options):
        data = self._data_dictionary[measure].reset_index()

        ax.scatter(
            data['date'], data['mean'],
            color=color_dict['light'], edgecolor=color_dict['dark'],
            alpha=self.observed_alpha
        )

        ax.plot(
            data['date'], data['mean'],
            color=color_dict['light'],
            alpha=self.observed_alpha / 2,
        )

    def format_date_axis(self, ax, start=None, end=None):
        start = start if start is not None else self._start
        end = end if end is not None else self._end
        date_locator = mdates.AutoDateLocator(maxticks=15)
        date_formatter = mdates.ConciseDateFormatter(date_locator, show_offset=False)
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.tick_params(axis='x', labelsize=self.tick_label_fontsize)

    def clean_and_align_axes(self, fig, axes):
        fig.align_ylabels(axes)
        # for ax in axes[:-1]:
        #     ax.xaxis.set_ticklabels([])
        
        
def write_or_show(fig, plot_file: Optional[Union[str, Path]]):
    if plot_file:
        fig.savefig(plot_file)
        plt.close(fig)
    else:
        plt.show()

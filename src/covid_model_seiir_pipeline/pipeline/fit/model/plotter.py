import itertools
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.lib.plotting import (
    Location,
    DataDict,
    Plotter,
    identity,
    pop_scale,
)

from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface

logger = cli_tools.task_performance_logger


class __PlotType(NamedTuple):
    model_fit: str
    model_fit_tail: str
    model_compare: str


PLOT_TYPE = __PlotType(*__PlotType._fields)


_COLOR_MEASURES = (
    ({'light': 'indianred', 'dark': 'darkred'}, ['death', 'deaths', 'ifr']),
    ({'light': 'mediumseagreen', 'dark': 'darkgreen'}, ['case', 'cases', 'idr']),
    ({'light': 'dodgerblue', 'dark': 'navy'}, ['admission', 'admissions', 'hospitalizations', 'ihr']),
)
MEASURE_COLORS = {
    measure: cmap for cmap, measures in _COLOR_MEASURES for measure in measures
}
VARIANT_COLORS = {
    'alpha': 'darkcyan',
    'beta': 'orangered',
    'gamma': 'saddlebrown',
    'delta': 'indigo',
    'omicron': 'crimson',
    'other': 'darkslategrey',
}


class PastPlotter(Plotter):
    fig_size = (30, 18)


def model_fit_plot(data: Tuple[Location, DataDict],
                   data_interface: FitDataInterface,
                   start: pd.Timestamp = pd.Timestamp(year=2020, month=2, day=1),
                   end: pd.Timestamp = pd.Timestamp.today(),
                   uncertainty: bool = False,
                   plot_root: str = None):
    location, data_dictionary = data
    assert len(data_dictionary) == 1, "Multiple versions supplied for model fit plot."
    version = list(data_dictionary)[0]  # This version name is arbitrary and not used anywhere
    pop = data_interface.load_population('total').population.loc[location.id]
    rhos = data_interface.load_variant_prevalence('reference')

    try:
        loc_rhos = rhos.loc[location.id].reset_index()
        variants = [v for v in loc_rhos if
                    v not in ['ancestral', 'omega', 'date'] and loc_rhos[v].max() > 0.5]
        variant_invasion = defaultdict(list)
        for v in variants:
            for threshold, linestyle in zip([0.01], ['-']):
                try:
                    variant_invasion[v].append(
                        (loc_rhos[loc_rhos[v] > threshold].date.iloc[0],
                        VARIANT_COLORS[v], linestyle)
                    )
                except IndexError:
                    variant_invasion[v].append((None, VARIANT_COLORS[v], linestyle))
    except KeyError:
        variant_invasion = {}

    # No versions so no specific style things here.
    style_map = {k: {} for k in data_dictionary}
    plotter = PastPlotter(
        data_dictionary=data_dictionary,
        version_style_map=style_map,
        location=location,
        start=start,
        end=end,
        uncertainty=uncertainty,
    )

    # Configure the plot layout.
    fig, grid_spec = plotter.build_grid_spec(
        nrows=1, ncols=3,
        width_ratios=[5, 3, 5],
        wspace=0.2,
    )
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
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        ## need naive (not naive-unvaccinated)
        # plotter.make_time_plot(
        #     ax=ax_measure,
        #     measure=f'posterior_daily_{measure}',
        #     color=MEASURE_COLORS[measure]['light'],
        #     linestyle=':',
        # )
        ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        plotter.make_observed_time_plot(
            ax=ax_measure,
            version=version,
            measure=f'daily_{measure}',
            light_color=MEASURE_COLORS[measure]['light'],
            dark_color=MEASURE_COLORS[measure]['dark'],
        )

        plotter.add_variant_vlines(ax_measure, variant_invasion)
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
        ax_measure = fig.add_subplot(gs_rates[2 * i])

        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'smoothed_cumulative_{measure}',
            color=MEASURE_COLORS[measure]['dark'],
            label=label,
        )

        plotter.make_observed_time_plot(
            ax=ax_measure,
            measure=f'cumulative_{measure}',
            version=version,
            light_color=MEASURE_COLORS[measure]['light'],
            dark_color=MEASURE_COLORS[measure]['dark'],
        )

        group_axes.append(ax_measure)

    rates_measures = [
        ('idr', (0, 100)),
        ('ihr', (0, 10)),
        ('ifr', (0, 5)),
    ]
    for i, (measure, ylim) in enumerate(rates_measures):
        ax_measure = fig.add_subplot(gs_rates[2 * i + 1])

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
        ratio_plot_range = data_dictionary[version][f'prior_{measure}']
        if ratio_plot_range is not None:
            ratio_plot_range = ratio_plot_range.loc[:, 'mean'] * 100

        rates_data = data_dictionary[version]['rates_data']
        if rates_data is not None:
            try:             
                rates_data = rates_data.loc[measure]
                ratio_plot_range = pd.concat([ratio_plot_range,
                                              rates_data.loc[:, 'value'].rename('mean') * 100])
                ax_measure.scatter(
                    rates_data.index,
                    rates_data.values * 100,
                    color=MEASURE_COLORS[measure]['dark'],
                    marker='o',
                    facecolors='none',
                    s=100,
                )
            except KeyError:
                pass

        if ratio_plot_range is not None:
            ratio_plot_range = (ratio_plot_range
                                .replace((-np.inf, np.inf), np.nan)
                                .dropna())
            ratio_plot_min, ratio_plot_max = ratio_plot_range.quantile((0, 1))
            ratio_plot_lim = (max(0, ratio_plot_min - ratio_plot_max * 0.2),
                              min(100, ratio_plot_max + ratio_plot_max * 0.2))
            ax_measure.set_ylim(ratio_plot_lim)
        else:
            ax_measure.set_ylim(ylim)

        group_axes.append(ax_measure)

    plotter.clean_and_align_axes(fig, group_axes)

    ax_daily = fig.add_subplot(gs_infecs[0])
    ax_cumul = fig.add_subplot(gs_infecs[1])

    for metric, ax, transform in [('daily', ax_daily, identity), ('cumulative', ax_cumul, pop_scale(pop))]:
        measure_type = 'naive' if metric == 'cumulative' else 'total'
        for measure in ['case', 'death', 'admission']:
            plotter.make_time_plot(
                ax=ax,
                measure=f'posterior_{measure}_based_{metric}_{measure_type}_infections',
                color=MEASURE_COLORS[measure]['light'],
                linestyle='--',
                transform=transform,
            )

        if metric == 'daily':
            plotter.make_time_plot(
                ax=ax,
                measure=f'posterior_{metric}_total_infections',
                color='black',
                label=f'{metric.capitalize()} Infections',
                uncertainty=True,
                transform=transform,
            )
            plotter.make_time_plot(
                ax=ax,
                measure=f'posterior_{metric}_naive_infections',
                color='black',
                transform=transform,
                linestyle=':',
            )
            plotter.add_variant_vlines(ax, variant_invasion)
        elif metric == 'cumulative':
            plotter.make_time_plot(
                ax=ax,
                measure=f'posterior_{metric}_naive_infections',
                color='black',
                label=f'{metric.capitalize()} Infected (%)',
                uncertainty=True,
                transform=transform,
            )

    sero_data = data_dictionary[version]['seroprevalence']
    if sero_data is not None:
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 1].index,
                         sero_data.loc[sero_data['is_outlier'] == 1, 'reported_seroprevalence'] * 100,
                         s=100, c='maroon', alpha=0.45, marker='x', zorder=2)
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 0].index,
                         sero_data.loc[sero_data['is_outlier'] == 0, 'reported_seroprevalence'] * 100,
                         s=100, c='darkturquoise', edgecolors='darkcyan', alpha=0.3, marker='s', zorder=2)
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 0].index,
                         sero_data.loc[sero_data['is_outlier'] == 0, 'seroprevalence'] * 100,
                         s=100, c='orange', edgecolors='darkorange', alpha=0.3, marker='^', zorder=2)
        ax_cumul.scatter(sero_data.loc[sero_data['is_outlier'] == 0].index,
                         sero_data.loc[sero_data['is_outlier'] == 0, 'adjusted_seroprevalence'] * 100,
                         s=150, c='mediumorchid', edgecolors='darkmagenta', alpha=0.6, marker='o', zorder=2)

    group_axes = [ax_daily, ax_cumul]
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
    ax_beta.set_ylim(-3, 2)
    plotter.add_variant_vlines(ax_beta, variant_invasion)

    group_axes.append(ax_beta)

    plotter.clean_and_align_axes(fig, group_axes)

    sns.despine(fig=fig, left=True, bottom=True)
    fig.suptitle(f'{location.name} ({location.id})', x=0.5, fontsize=plotter.title_fontsize, ha='center')

    if plot_root:
        plot_path = Path(plot_root) / f'ies_{location.id}.pdf'
    else:
        plot_path = None
    plotter.write_or_show(fig, plot_path)


def model_compare_plot(data: Tuple[Location, DataDict],
                       data_interface: FitDataInterface,
                       start: pd.Timestamp = pd.Timestamp(year=2020, month=2, day=1),
                       end: pd.Timestamp = pd.Timestamp.today(),
                       uncertainty: bool = False,
                       plot_root: str = None):
    location, data_dictionary = data
    assert len(data_dictionary) == 4, "Incorrect number of versions supplied for model comparison plot."
    # Assume v1 round 1, v1 round 2, v2 round1, v2 round2 ordering
    style_map = {version: {'color': color, 'linestyle': line_style} for version, (color, line_style)
                 in zip(data_dictionary, itertools.product(['#7F3C8D', '#11A579'], ['solid', 'dashed']))}
    plotter = PastPlotter(
        data_dictionary=data_dictionary,
        version_style_map=style_map,
        location=location,
        start=start,
        end=end,
        uncertainty=uncertainty,
    )

    # Configure the plot layout.
    fig, grid_spec = plotter.build_grid_spec(
        nrows=1, ncols=5,
        wspace=0.2,
    )
    gs_daily = grid_spec[0, 0].subgridspec(3, 1)
    gs_rates_prior = grid_spec[0, 1].subgridspec(3, 1)
    gs_rates = grid_spec[0, 2].subgridspec(3, 1)
    gs_beta = grid_spec[0, 3].subgridspec(3, 1)
    gs_infecs = grid_spec[0, 4].subgridspec(3, 1)

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
            measure=f'posterior_daily_{measure}',
            label=label,
        )
        group_axes.append(ax_measure)

    plotter.clean_and_align_axes(fig, group_axes)

    # Column 2, Cumulative & rates
    group_axes = []
    rates_measures = [
        ('idr', (0, 100)),
        ('ihr', (0, 10)),
        ('ifr', (0, 5)),
    ]
    for i, (measure, ylim) in enumerate(rates_measures):
        ax_measure = fig.add_subplot(gs_rates_prior[i])
        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'prior_{measure}',
            transform=lambda x: x * 100,
            label=f'Prior {measure.upper()} (%)',
        )
        ax_measure.set_ylim(ylim)

        ax_measure = fig.add_subplot(gs_rates[i])
        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'posterior_{measure}',
            transform=lambda x: x * 100,
            label=f'Posterior {measure.upper()} (%)',
        )
        ax_measure.set_ylim(ylim)

        group_axes.append(ax_measure)

    plotter.clean_and_align_axes(fig, group_axes)

    group_axes = []
    for i, measure in enumerate(['cases', 'hospitalizations', 'deaths']):
        ax_beta = fig.add_subplot(gs_beta[i])
        plotter.make_time_plot(
            ax=ax_beta,
            measure=f'beta_{measure}',
            label=f'Log Beta {measure.capitalize()}',
            transform=np.log,
        )
        ax_beta.set_ylim(-3, 2)

        group_axes.append(ax_beta)

    plotter.clean_and_align_axes(fig, group_axes)

    group_axes = []

    for i, group in enumerate(['naive', 'total']):
        ax = fig.add_subplot(gs_infecs[i])
        plotter.make_time_plot(
            ax=ax,
            measure=f'posterior_daily_{group}_infections',
            label=f'Daily {(" ".join(group.split("_"))).capitalize()} Infections',
        )
        group_axes.append(ax)

    ax_beta = fig.add_subplot(gs_infecs[2])
    plotter.make_time_plot(
        ax=ax_beta,
        measure=f'beta',
        label=f'Log Beta',
        transform=np.log,
    )
    ax_beta.set_ylim(-3, 2)
    group_axes.append(ax_beta)
    plotter.clean_and_align_axes(fig, group_axes)
    plotter.despine_and_make_title(fig)
    plotter.make_legend(fig)

    if plot_root:
        plot_path = Path(plot_root) / f'compare_{location.id}.pdf'
    else:
        plot_path = None
    plotter.write_or_show(fig, plot_path)

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.lib.plotting import (
    Location,
    Plotter,
    identity,
    pop_scale,
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


class ModelFitPlotter(Plotter):
    fig_size = (30, 18)


def model_fit_plot(data: Tuple[Location, Dict[str, Dict[str, pd.DataFrame]]],
                   data_interface: FitDataInterface,
                   start: pd.Timestamp = pd.Timestamp(year=2020, month=2, day=1),
                   end: pd.Timestamp = pd.Timestamp.today(),
                   uncertainty: bool = False,
                   review: bool = False,
                   plot_root: str = None):
    location, data_dictionary = data
    assert len(data_dictionary) == 1, "Multiple versions supplied for model fit plot."
    version = list(data_dictionary)[0]  # This version name is arbitrary and not used anywhere
    pop = data_interface.load_population('total').population.loc[location.id]

    # No versions so no specific style things here.
    style_map = {k: {} for k in data_dictionary}
    plotter = ModelFitPlotter(
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
        plotter.make_time_plot(
            ax=ax_measure,
            measure=f'posterior_daily_{measure}',
            color=MEASURE_COLORS[measure]['light'],
            linestyle='--',
        )
        plotter.make_observed_time_plot(
            ax=ax_measure,
            version=version,
            measure=f'daily_{measure}',
            light_color=MEASURE_COLORS[measure]['light'],
            dark_color=MEASURE_COLORS[measure]['dark'],
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
        ax_measure = fig.add_subplot(gs_rates[2 * i])

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
                    s=100
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
            label=f'{metric.capitalize()} Infected{suffix}',
            uncertainty=True,
            transform=transform,
        )
        if metric == 'daily':
            plotter.make_time_plot(
                ax=ax,
                measure=f'posterior_{metric}_total_infections',
                color='black',
                transform=transform,
                linestyle=':',
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

    if review:
        dd = data_dictionary[version]
        ax_stackplot = fig.add_subplot(gs_infecs[2])
        naive_unvax = dd['posterior_cumulative_naive_unvaccinated_infections'].loc[:, 'mean'] * 100 / pop
        naive = dd['posterior_cumulative_naive_infections'].loc[:, 'mean'] * 100 / pop
        total = dd['posterior_cumulative_total_infections'].loc[:, 'mean'] * 100 / pop
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
        ax_beta.set_ylim(-3, 2)

        group_axes.append(ax_beta)

    plotter.clean_and_align_axes(fig, group_axes)

    sns.despine(fig=fig, left=True, bottom=True)
    fig.suptitle(f'{location.name} ({location.id})', x=0.5, fontsize=plotter.title_fontsize, ha='center')

    if plot_root:
        plot_path = Path(plot_root) / f'ies_{location.id}.pdf'
    else:
        plot_path = None
    plotter.write_or_show(fig, plot_path)

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)

logger = cli_tools.task_performance_logger


class Location(NamedTuple):
    id: int
    name: str


DataDict = Dict[str, Dict[str, pd.DataFrame]]


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
    line_width = 2.5
    grid_spec_margins = {'top': 0.92, 'bottom': 0.08}

    def __init__(self,
                 data_dictionary: DataDict,
                 version_style_map: Dict[str, Dict[str, str]],
                 location: Location,
                 start: pd.Timestamp,
                 end: pd.Timestamp,
                 uncertainty: bool,
                 transform=identity,
                 **extra_defaults):
        sns.set_style('whitegrid')
        self._data_dictionary = data_dictionary
        self._version_style_map = version_style_map

        self._location = location
        self._start = start
        self._end = end

        self._uncertainty = uncertainty
        self._transform = transform
        self._default_options = {
            'linewidth': self.line_width,
            'linestyle': 'solid',
            **extra_defaults
        }

    def make_time_plot(self,
                       ax: Axes,
                       measure: str,
                       label: str = None,
                       **extra_options) -> None:
        uncertainty = extra_options.pop('uncertainty', self._uncertainty)
        transform = extra_options.pop('transform', self._transform)
        start = extra_options.pop('start', self._start)
        end = extra_options.pop('end', self._end)

        for version, version_options in self._version_style_map.items():
            # This configuration is hierarchical. extra overrides version overrides default.
            plot_options = {**self._default_options, **version_options, **extra_options}
            assert 'color' in plot_options
            data = self._data_dictionary[version][measure]
            if data is None or np.all(data['mean'] == 0):
                return
            data = data.reset_index()
            data = data[(start <= data['date']) & (data['date'] <= end)]

            transform_cols = [c for c in data if c in ['mean', 'upper', 'lower', 'upper2', 'lower2']]
            data[transform_cols] = transform(data[transform_cols])

            ax.plot(data['date'], data['mean'], **plot_options)
            if uncertainty:
                ax.fill_between(data['date'], data['upper'], data['lower'],
                                alpha=self.fill_alpha, color=plot_options['color'])
                if 'upper2' in data:
                    ax.fill_between(data['date'], data['upper2'], data['lower2'],
                                    alpha=1.5 * self.fill_alpha, color=plot_options['color'])

        self.format_date_axis(ax, start, end)

        if label is not None:
            ax.set_ylabel(label, fontsize=self.ax_label_fontsize)
            ax.tick_params(axis='y', labelsize=self.tick_label_fontsize)

    def make_observed_time_plot(self,
                                ax: Axes,
                                version: str,
                                measure: str,
                                light_color: str,
                                dark_color: str,
                                **extra_options) -> None:
        data = self._data_dictionary[version][measure].reset_index()
        ax.scatter(
            data['date'], data['mean'],
            color=light_color,
            edgecolor=dark_color,
            alpha=self.observed_alpha
        )

        ax.plot(
            data['date'], data['mean'],
            color=light_color,
            alpha=self.observed_alpha / 2,
            **extra_options,
        )

    def add_variant_vlines(self,
                           ax: Axes,
                           variant_spec: Dict[str, List[Tuple[Optional[pd.Timestamp], str, str]]]):
        for variant, invasion_level_spec in variant_spec.items():
            trans = ax.get_xaxis_transform()
            for i, (date, color, linestyle) in enumerate(invasion_level_spec):
                if date is not None:
                    ax.axvline(date, linestyle=linestyle, color=color, linewidth=3)
                    if i == 0:
                        ax.text(date, 0.7, variant,
                                transform=trans, rotation=90, fontsize=14, color=color)

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

    def despine_and_make_title(self, fig):
        sns.despine(fig=fig, left=True, bottom=True)
        fig.suptitle(f'{self._location.name} ({self._location.id})',
                     x=0.5,
                     fontsize=self.title_fontsize,
                     ha='center')

    def make_legend(self, fig):
        fig.legend(
            handles=self.make_legend_handles(),
            loc='lower center',
            bbox_to_anchor=(0.5, 0),
            fontsize=self.ax_label_fontsize,
            frameon=False,
            ncol=len(self._version_style_map)
        )

    def make_legend_handles(self):
        handles = []
        for label, line_properties in self._version_style_map.items():
            handles.append(mlines.Line2D(
                [], [],
                label=label.title(),
                linewidth=self.line_width,
                **line_properties,
            ))
        return handles

    def build_grid_spec(self, **grid_spec_args):
        fig = plt.figure(figsize=self.fig_size, tight_layout=True)
        grid_spec = fig.add_gridspec(
            **grid_spec_args
        )
        grid_spec.update(**self.grid_spec_margins)
        return fig, grid_spec

    @staticmethod
    def write_or_show(fig, plot_file: Optional[Union[str, Path]]) -> None:
        if plot_file:
            fig.savefig(plot_file)
            plt.close(fig)
        else:
            plt.show()

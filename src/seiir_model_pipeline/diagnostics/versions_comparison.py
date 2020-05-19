import matplotlib

import os
import numpy as np
import pandas as pd

from typing import List
from matplotlib import pyplot as plt

from seiir_model_pipeline.diagnostics.visualizer import Visualizer
from seiir_model_pipeline.core.versioner import Directories
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.data import load_covariates

matplotlib.use('Agg')


class VersionsComparator:

    def __init__(self, list_of_directories: List[Directories],
                 groups: list = None, exclude_groups: list = None,
                 col_group="loc_id", col_date='date', col_observed='observed',
                 covariates=()):
        self.visualizers = {}
        self.groups = groups
        # TODO: Need some other way to get the list of default covariates
        self.default_covariates = [
            "intercept", "mobility_lift", "proportion_over_1k", "temperature", "testing_reference"
        ]
        if exclude_groups is not None:
            for exclude_group in exclude_groups:
                self.groups.remove(exclude_group)

        for directories in list_of_directories:
            self.visualizers[directories.regression_version] = Visualizer(
                directories,
                groups=groups,
                exclude_groups=exclude_groups,
                col_group=col_group,
                col_date=col_date,
                col_observed=col_observed,
                covariates=covariates,
                read_coefficients_draws=True,
            )

            self.visualizers[directories.regression_version].read_coefficient_draws()

    def compare_coefficients_by_location_plot(self, group, covariates=None, output_dir=".", base_fig_size=(2.5, 5)):
        """
        Box plots for comparing locations coefficients for different versions
        """
        assert len(self.visualizers) != 0, "No visualizers were provided. Check the list_of_directories in constructor"

        versions = [k for k, v in self.visualizers.items()]
        if covariates is None:
            covariates = self.default_covariates

        group_name = self.visualizers[versions[0]].id2loc[group]

        fig, ax = plt.subplots(len(covariates), 1,
                               figsize=(base_fig_size[0] * len(versions), base_fig_size[1] * len(covariates)))

        for i, covariate in enumerate(covariates):
            covariate_all_values = []
            for version, visualizer in self.visualizers.items():
                draws = visualizer.coefficients_draws
                covariate_values = []
                for draw in draws:
                    covariate_values.append(draw.loc[draw.group_id == group, covariate].iloc[0])
                covariate_all_values.append(covariate_values)
            ax[i].set_title(f"{covariate} for {group_name}")
            ax[i].boxplot(covariate_all_values, labels=versions, notch=True, widths=0.4)
        plt.savefig(os.path.join(output_dir, f"regression_coefs_comparison_{group_name}.pdf"))
        plt.close(fig)

    def compare_coefficients_scatterplot(self, covariates=None, base_fig_size=(6, 6), output_dir="."):
        """
        Scatter plots for coefficients
        """
        if covariates is None:
            covariates = self.default_covariates

        fig = plt.figure(figsize=(base_fig_size[0] * 2, base_fig_size[1] * len(covariates)))
        grid = plt.GridSpec(len(covariates), 2, wspace=0.3, hspace=0.3)

        assert len(self.visualizers) == 2  # plots are version1 vs version2

        versions = [k for k, v in self.visualizers]
        cov_plots = []

        for i, covariate in enumerate(covariates):
            cov_plot = fig.add_subplot(grid[i // 2, i % 2])
            cov_plot.set_title(covariate)
            cov_plot.set_xlabel(versions[0])
            cov_plot.set_ylabel(versions[1])
            cov_plots.append(cov_plot)

        draws1 = self.visualizers[versions[0]].coefficients_draws
        draws2 = self.visualizers[versions[1]].coefficients_draws

        for j, (draw1, draw2) in enumerate(zip(draws1, draws2)):
            draw2.rename(columns={name: name + "_2" for name in draw2.columns if name != "group_id"}, inplace=True)
            data = draw1.merge(draw2, how="inner", on="group_id")
            for i, (covariate, plot) in enumerate(zip(covariates, cov_plots)):
                plot.scatter(data[covariate], data[covariate + "_2"], alpha=0.1)

        plt.savefig(os.path.join(output_dir, f"all_locs_reg_coefs_comparison.png"))
        plt.close(fig)

    @staticmethod
    def compare_input_covariates(regression_versions, groups, base_fig_size=(12, 6), output_dir=".",
                                 colors=('r', 'b')):
        assert len(regression_versions) == 2, f"This function compares two versions, " \
                                              f" but {len(regression_versions)} were provided"
        cov_data = {}
        id2loc = None
        covariates = None
        for version in regression_versions:
            settings = load_regression_settings(version)
            directories = Directories(forecast_version=version)
            cov_data[version] = load_covariates(directories=directories,
                                                covariate_version=settings.covariate_cache_version,
                                                location_ids=groups)
            if id2loc is None:
                location_metadata = pd.read_csv(
                    directories.get_location_metadata_file(
                        settings.location_set_version_id)
                )
                id2loc = location_metadata.set_index('location_id')[
                    'location_name'].to_dict()
            if covariates is None:
                covariates = set(settings.covariates.keys())
            else:
                covariates = covariates & set(settings.covariates.keys())

        for group in groups:
            group_name = id2loc[group]
            fig = plt.figure(figsize=(base_fig_size[0], base_fig_size[1] * len(covariates)))
            grid = plt.GridSpec(len(covariates), 1)
            something_changed = False

            for i, covariate in enumerate(covariates):
                cov_plot = fig.add_subplot(grid[i, 0])
                values = []
                title = f"{covariate} for {group_name}: "
                data = [cov_data[regression_versions[i]][cov_data[regression_versions[i]]["location_id"] == group] for i
                        in range(2)]

                if len(data[0][covariate].unique()) == len(data[1][covariate].unique()) == 1:
                    cov1 = data[0][covariate].iloc[0]
                    cov2 = data[1][covariate].iloc[0]
                    cov_plot.scatter(cov1, cov2, label=covariate + (" CHANGED" if not np.allclose(cov1, cov2) else ""))
                    values += [cov1, cov2]
                    xlims = cov_plot.get_xlim()
                    ylims = cov_plot.get_ylim()
                    xmin = min(xlims[0], ylims[0])
                    xmax = max(xlims[1], ylims[1])
                    cov_plot.plot([xmin, xmax], [xmin, xmax], "--", c='black', label="diagonal")
                    cov_plot.set_xlim((xmin, xmax))
                    cov_plot.set_ylim((xmin, xmax))
                    this_covariate_changed = not np.allclose(cov1, cov2)

                else:
                    for j, df in enumerate(data):
                        time = pd.to_datetime(df["date"])
                        cov_plot.plot(time, df[covariate], c=colors[j], label=regression_versions[j])
                        values.append(df[covariate].to_numpy())
                    this_covariate_changed = not (
                            len(values[0]) == len(values[1]) and np.allclose(values[0], values[1], equal_nan=True))

                cov_plot.set_title(title + ("CHANGED" if this_covariate_changed else "not changed"))
                cov_plot.legend()
                something_changed = something_changed or this_covariate_changed

            plt.savefig(os.path.join(output_dir,
                                     (("CHANGED " if something_changed else "")
                                      + f"{group_name}_covariates_comparison.jpg")))
            plt.close(fig)

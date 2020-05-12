import os
from typing import List

from matplotlib import pyplot as plt

from src.seiir_model_pipeline.diagnostics.visualizer import Visualizer
from seiir_model_pipeline.core.versioner import Directories


class VersionsComparator:

    def __init__(self, list_of_directories: List[Directories],
                 groups: list = None, exclude_groups: list = None,
                 col_group="loc_id", col_date='date', col_observed='observed',
                 covariates=()):
        self.visualizers = {}
        self.groups = groups
        if exclude_groups is not None:
            for exclude_group in exclude_groups:
                self.groups.remove(exclude_group)

        for directories in list_of_directories:
            self.visualizers[directories.regression_version] = Visualizer(directories,
                                                                          groups=groups,
                                                                          exclude_groups=exclude_groups,
                                                                          col_group=col_group,
                                                                          col_date=col_date,
                                                                          col_observed=col_observed,
                                                                          covariates=covariates,
                                                                          read_coefficients_draws=True,
                                                                          )

    def compare_coefficients_plot(self, group, covariates=None, output_dir=".", base_fig_size=(2.5, 5)):
        assert len(self.visualizers) != 0, "No visualizers were provided. Check the list_of_directories in constructor"

        versions = [k for k, v in self.visualizers]

        if covariates is None:
            # TODO: Need some other way to get the list of default covariates
            covariates = ["intercept", "mobility_lift", "proportion_over_1k", "temperature", "testing_reference"]

        group_name = self.visualizers[versions[0]].id2loc[group]

        fig = plt.figure(figsize=(base_fig_size[0] * len(versions), base_fig_size[1] * len(covariates)))
        grid = plt.GridSpec(len(covariates), 1)

        for i, covariate in enumerate(covariates):
            covariate_all_values = []
            for version, visualizer in self.visualizers.items():
                draws = visualizer.coefficients_draws
                covariate_values = []
                for draw in draws:
                    covariate_values.append(draw.loc[group, covariate])
                covariate_all_values.append(covariate_values)
            cov_plot = fig.add_subplot(grid[i, 0])
            cov_plot.set_title(f"{covariate} for {group_name}")
            cov_plot.boxplot(covariate_all_values, labels=versions, notch=True, widths=0.4)
            cov_plot.grid()
            plt.savefig(os.path.join(output_dir, f"regression_coefs_comparison_{group_name}.png"))

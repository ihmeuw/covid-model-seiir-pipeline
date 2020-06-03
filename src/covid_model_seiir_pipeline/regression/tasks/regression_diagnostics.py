from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from typing import Optional
import shlex


import numpy as np
import pandas as pd

import matplotlib

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface

log = logging.getLogger(__name__)

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PlotBetaCoef:

    def __init__(self, regression_version: str):
        # load specifications
        regress_spec: RegressionSpecification = RegressionSpecification.from_path(
            Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
        )
        self.regression_specification = regress_spec
        ode_fit_spec: FitSpecification = FitSpecification.from_path(
            Path(self.regression_specification.data.ode_fit_version) /
            static_vars.FIT_SPECIFICATION_FILE
        )

        # data interface
        self.data_interface = RegressionDataInterface(
            regression_root=Path(self.regression_specification.data.output_root),
            covariate_root=Path(self.regression_specification.data.covariate_version),
            ode_fit_root=Path(ode_fit_spec.data.output_root),
            infection_root=Path(ode_fit_spec.data.infection_version),
            location_file=(Path('/ihme/covid-19/seir-pipeline-outputs/metadata-inputs') /
                           f'location_metadata_{ode_fit_spec.data.location_set_version_id}.csv')
        )
        self.path_to_savefig = self.data_interface.regression_paths.diagnostic_dir

        # load metadata
        location_metadata = pd.read_csv(self.data_interface.location_metadata_file)
        self.id2loc = location_metadata.set_index('location_id')['location_name'].to_dict()

        # organize information
        self.covs = np.sort(list(self.regression_specification.covariates.keys()))

        # load coef
        df_coef = [
            self.data_interface.load_mr_coefficients(i)
            for i in range(ode_fit_spec.parameters.n_draws)
        ]
        self.loc_ids = np.sort(list(df_coef[0]['group_id'].unique()))
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])
        self.num_locs = len(self.locs)

        # group coef data
        self.coef_data = {}
        for cov in self.covs:
            coef_mat = np.vstack([
                df[cov].values
                for df in df_coef
            ])
            coef_label = self.locs.copy()
            coef_mean = coef_mat.mean(axis=0)
            sort_idx = np.argsort(coef_mean)
            self.coef_data[cov] = (coef_label[sort_idx], coef_mat[:, sort_idx])

    def plot_coef(self):
        for cov in self.covs:
            plt.figure(figsize=(8, self.coef_data[cov][1].shape[1]//4))
            plt.boxplot(self.coef_data[cov][1], vert=False, showfliers=False,
                        boxprops=dict(linewidth=0.5),
                        whiskerprops=dict(linewidth=0.5))
            plt.yticks(ticks=np.arange(self.num_locs) + 1,
                       labels=self.coef_data[cov][0])

            coef_mean = self.coef_data[cov][1].mean()
            plt.vlines(coef_mean, ymin=1, ymax=self.num_locs,
                       linewidth=1.0, linestyle='--', color='#003152')

            plt.grid(b=True)
            plt.box(on=None)
            plt.title(cov)
            plt.savefig(self.path_to_savefig / f'{cov}_boxplot.pdf',
                        bbox_inches='tight')

        # save the coefficient of stats
        for cov in self.covs:
            lower = np.quantile(self.coef_data[cov][1], 0.025, axis=0)
            upper = np.quantile(self.coef_data[cov][1], 0.975, axis=0)
            mean = np.mean(self.coef_data[cov][1], axis=0)
            df = pd.DataFrame({
                'loc': self.locs,
                'loc_id': self.loc_ids,
                'lower': lower,
                'mean': mean,
                'upper': upper,
            })
            df.to_csv(self.path_to_savefig/f'{cov}_coef.csv', index=False)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--regression-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    log.info("Initiating SEIIR diagnostics.")

    handle = PlotBetaCoef(regression_version=args.regression_version)
    handle.plot_coef()


if __name__ == '__main__':
    main()

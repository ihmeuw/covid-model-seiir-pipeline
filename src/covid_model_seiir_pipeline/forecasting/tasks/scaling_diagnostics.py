from argparse import ArgumentParser, Namespace
import logging
import os
from pathlib import Path
from typing import Optional
import shlex

import numpy as np
import pandas as pd
import matplotlib

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification


log = logging.getLogger(__name__)

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PlotBetaScaling:

    def __init__(self, forecast_version: str, scenario: str):
        forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
            Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
        )
        regress_spec: RegressionSpecification = RegressionSpecification.from_path(
            Path(forecast_spec.data.regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
        )
        ode_fit_spec: FitSpecification = FitSpecification.from_path(
            Path(regress_spec.data.ode_fit_version) / static_vars.FIT_SPECIFICATION_FILE
        )
        self.forecast_specification = forecast_spec

        self.data_interface = ForecastDataInterface(
            forecast_root=Path(forecast_spec.data.output_root) / scenario,
            regression_root=Path(regress_spec.data.output_root),
            ode_fit_root=Path(ode_fit_spec.data.output_root),
            infection_root=Path(ode_fit_spec.data.infection_version),
            location_file=Path(ode_fit_spec.data.location_set_file)
        )

        # load settings
        self.path_to_savefig = self.data_interface.forecast_paths.diagnostics

        # load location metadata
        location_metadata = pd.read_csv(self.data_interface.location_metadata_file)
        self.id2loc = location_metadata.set_index('location_id')['location_name'].to_dict()

        # load locations
        self.loc_ids = np.array([
            file_name.split('_')[0]
            for file_name in os.listdir(self.data_interface.forecast_paths.beta_scaling)
        ]).astype(int)
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])

        # load data
        self.scales_data = np.hstack([
            self.data_interface.load_beta_scales(loc_id)['beta_scales'].values[:, None]
            for loc_id in self.loc_ids
        ])

    def plot_scales(self):
        scales_mean = self.scales_data.mean(axis=0)
        sort_id = np.argsort(scales_mean)
        plt.figure(figsize=(8, len(self.locs)//4))
        plt.boxplot(self.scales_data[:, sort_id], vert=False, showfliers=False,
                    boxprops=dict(linewidth=0.5),
                    whiskerprops=dict(linewidth=0.5))
        plt.yticks(ticks=np.arange(len(self.locs)) + 1,
                   labels=self.locs[sort_id])
        plt.grid(b=True)
        plt.box(on=None)
        plt.title('beta scalings')
        plt.vlines(1.0, ymin=1, ymax=len(self.locs),
                   linewidth=1.0, linestyle='--', color='#8B0000')
        plt.savefig(self.path_to_savefig / f'beta_scalings_boxplot.pdf',
                    bbox_inches='tight')


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    log.info("Initiating SEIIR diagnostics.")

    handle = PlotBetaScaling(forecast_version=args.forecast_version,
                             scenario=args.scenario)
    handle.plot_scales()


if __name__ == '__main__':
    main()

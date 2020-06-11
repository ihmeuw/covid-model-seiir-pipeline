from argparse import ArgumentParser, Namespace
import logging
import os
from pathlib import Path
import shlex
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd

import matplotlib

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface


log = logging.getLogger(__name__)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


ODE_BETA_FIT = "ode_beta_fit"
COEFFICIENTS_FIT = "coefficients_fit"
PARAMETERS_FIT = "param/usr/local/bin/lolcateters_fit"
ODE_COMPONENTS_FORECAST = "ode_forecast"
OUTPUT_DRAWS_CASES = "output_draws_cases"
OUTPUT_DRAWS_DEATHS = "output_draws_deaths"
OUTPUT_DRAWS_REFF = "output_draws_reff"


class Visualizer:

    def __init__(self, forecast_version: str, scenario: str, groups: Optional[list] = None,
                 exclude_groups: Optional[list] = None, col_group: str = "loc_id",
                 col_date: str = 'date', col_observed: str = 'observed', covariates: tuple = ()
                 ) -> None:

        # load settings
        forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
            Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
        )
        self.forecast_specification = forecast_spec
        regress_spec: RegressionSpecification = RegressionSpecification.from_path(
            Path(forecast_spec.data.regression_version) /
            static_vars.REGRESSION_SPECIFICATION_FILE
        )
        self.regression_specification = regress_spec
        ode_fit_spec: FitSpecification = FitSpecification.from_path(
            Path(regress_spec.data.ode_fit_version) / static_vars.FIT_SPECIFICATION_FILE
        )
        self.ode_fit_specification = ode_fit_spec

        # column names
        self.col_group = col_group
        self.col_date = col_date
        self.col_observed = col_observed

        # column subset
        if groups is None:
            groups = []
        self.groups = groups
        if exclude_groups is not None:
            for exclude_group in exclude_groups:
                self.groups.remove(exclude_group)

        # data containers
        self.data: Dict[str, Dict[str, Any]] = {
            group: {
                ODE_BETA_FIT: [],
                COEFFICIENTS_FIT: [],
                ODE_COMPONENTS_FORECAST: [],
                OUTPUT_DRAWS_CASES: None,
                OUTPUT_DRAWS_DEATHS: None,
                OUTPUT_DRAWS_REFF: None
            } for group in self.groups
        }
        self.params_for_draws: List = []
        self.covariates: Dict[str, pd.DataFrame] = {}

        # data interfaces
        regression_data_interface = RegressionDataInterface(
            regression_root=Path(self.regression_specification.data.output_root),
            covariate_root=Path(self.regression_specification.data.covariate_version),
            ode_fit_root=Path(ode_fit_spec.data.output_root),
            infection_root=Path(ode_fit_spec.data.infection_version),
            location_file=Path(ode_fit_spec.data.location_set_file)
        )
        self.forecast_data_interface = ForecastDataInterface(
            forecast_root=Path(self.forecast_specification.data.output_root) / scenario,
            regression_root=Path(self.regression_specification.data.output_root),
            ode_fit_root=Path(ode_fit_spec.data.output_root),
            infection_root=Path(ode_fit_spec.data.infection_version),
            location_file=Path(ode_fit_spec.data.location_set_file)
        )
        # self.path_to_savefig = self.data_interface.regression_paths.diagnostic_dir

        # load metadata
        location_metadata = pd.read_csv(regression_data_interface.location_metadata_file)
        self.id2loc = location_metadata.set_index('location_id')['location_name'].to_dict()

        # read beta regression draws
        for group in groups:
            for draw in range(self.ode_fit_specification.parameters.n_draws):
                draw_df = regression_data_interface.load_regression_betas(group, draw)
                # It's assumed that draw_df contains only the `group` group exclusively
                self.data[group][ODE_BETA_FIT].append(draw_df)

        # read components forecast
        for group in groups:
            for draw in range(self.ode_fit_specification.parameters.n_draws):
                draw_df = self.forecast_data_interface.load_component_forecasts(
                    location_id=group, draw_id=draw
                )
                self.data[group][ODE_COMPONENTS_FORECAST].append(draw_df)

        #  read final draws
        for group in groups:
            cases_df = self.forecast_data_interface.load_cases(group)
            deaths_df = self.forecast_data_interface.load_deaths(group)
            reff_df = self.forecast_data_interface.load_reff(group)

            self.data[group][OUTPUT_DRAWS_CASES] = cases_df
            self.data[group][OUTPUT_DRAWS_DEATHS] = deaths_df
            self.data[group][OUTPUT_DRAWS_REFF] = reff_df

    @staticmethod
    def format_x_axis(ax, start_date, now_date, end_date,
                      major_tick_interval_days=7, margins_days=5):

        months = mdates.DayLocator(interval=major_tick_interval_days)
        days = mdates.DayLocator()  # Every day
        months_fmt = mdates.DateFormatter('%m/%d')

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        ax.xaxis.set_minor_locator(days)

        # round to nearest years.
        datemin = np.datetime64(start_date, 'D') - np.timedelta64(margins_days, 'D')
        datemax = np.datetime64(end_date, 'D') + np.timedelta64(margins_days, 'D')
        ax.set_xlim(datemin, datemax)

        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

        if now_date is not None:
            now_date = np.datetime64(now_date, 'D')
            ylims = ax.get_ylim()
            ax.plot([now_date, now_date], ylims, linestyle="dashed", c='black')
            label_level = 0.9 * ylims[0] + 0.1 * ylims[1]
            ax.text(now_date - np.timedelta64(8, 'D'), label_level, "Past")
            ax.text(now_date + np.timedelta64(2, 'D'), label_level, "Future")

    def plot_ode_compartment(self, group, ax, compartment="I1",
                             linestyle='solid', transparency=0.2, color='orange', draws=None):

        for i, (past_compartments, future_compartments) in enumerate(zip(self.data[group][ODE_BETA_FIT],
                                                                         self.data[group][ODE_COMPONENTS_FORECAST])):
            if draws is not None:
                if i not in draws:
                    continue

            past_time = pd.to_datetime(past_compartments[self.col_date])
            past_compartment_trajectory = past_compartments[compartment]
            ax.plot(past_time, past_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)
            future_time = pd.to_datetime(future_compartments[self.col_date])
            future_compartment_trajectory = future_compartments[compartment]
            ax.plot(future_time, future_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)

        # get times
        past_time = pd.to_datetime(self.data[group][ODE_BETA_FIT][0][self.col_date])
        future_time = pd.to_datetime(self.data[group][ODE_COMPONENTS_FORECAST][0][self.col_date])
        start_date = past_time.to_list()[0]
        now_date = past_time.to_list()[-1]
        end_date = future_time.to_list()[-1]

        self.format_x_axis(ax, start_date, now_date, end_date, major_tick_interval_days=14)

    def plot_spline(self):
        num_locs = len(self.data)
        fig, ax = plt.subplots(num_locs, 1,
                               figsize=(8, 4 * num_locs))
        for i, loc_id in enumerate(self.data.keys()):
            df = self.data[loc_id][ODE_BETA_FIT].sort_values('date')
            for j in range(self.regression_settings.n_draws):
                ax[i].scatter(np.arange(df.shape[0]), df['newE_obs'],
                              marker='.', c='#ADD8E6', alpha=0.7)
                ax[i].plot(np.arange(df.shape[0]), df['newE'], c='#008080',
                           alpha=0.5, linewidth=1.0)
            ax[i].set_xticks(np.arange(df.shape[0])[::5])
            ax[i].set_xticklabels(df['date'][::5], rotation='vertical')
            ax[i].set_title()

        plt.savefig(self.directories.regression_diagnostic_dir / 'spline_fit.png',
                    bbox_inches='tight')

    def create_trajectories_plot(self, group, compartments=('S', 'E', 'I1', 'I2', 'R', 'beta'),
                                 colors=('blue', 'orange', 'red', 'purple', 'green', 'blue')):

        group_name = self.id2loc[group]

        fig = plt.figure(figsize=(12, (len(compartments) + 1) * 6))
        grid = plt.GridSpec(len(compartments) + 1, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()
        for i, compartment in enumerate(compartments):
            ax = fig.add_subplot(grid[i, 0])
            self.plot_ode_compartment(
                group=group, ax=ax,
                compartment=compartment,
                linestyle="solid",
                transparency=0.1,
                color=colors[i],
            )
            ax.grid(True)
            ax.set_title(f"Location {group_name}: {compartment}")

        print(f"Trajectories plot for {group} {group_name} is done")

        plt.savefig(
            self.forecast_data_interface.forecast_paths.get_trajectory_plots(group_name)
        )
        plt.close(fig)

    def create_beta_fit_and_residuals_plot(self, group):
        group_name = self.id2loc[group]
        fig = plt.figure(figsize=(12, 2 * 6))
        grid = plt.GridSpec(2, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()
        E_plot = fig.add_subplot(grid[0, 0])
        E_plot.set_title(f"Cases vs Cases Spline Fit for {group_name}")
        residuals_plot = fig.add_subplot(grid[1, 0])
        residuals_plot.set_title(f"Log-residuals for Beta Fit for {group_name}")
        residuals_plot.set_ylabel(f"log(beta)-log(beta_pred)")
        time = None
        for i, draw in enumerate(self.data[group][ODE_BETA_FIT]):
            time = pd.to_datetime(draw[self.col_date])
            E_plot.plot(time, draw['newE'], c='b', alpha=0.1, label="Spline Fit" if i == 0 else None)
            E_plot.scatter(time, draw['newE_obs'], c='b', alpha=0.1, s=3, label="Observations" if i == 0 else None)
            residuals = np.log(draw['beta'].to_numpy()) - np.log(draw['beta_pred'].to_numpy())
            residuals_plot.plot(time, residuals, c='b', alpha=0.1)

        if time is None:
            # No draws => no picture
            plt.close(fig)
            return None

        # Assuming the time is (almost) the same for all draws:
        start_date = time.to_list()[0]
        end_date = time.to_list()[-1]

        self.format_x_axis(E_plot,
                           start_date=start_date,
                           now_date=None,  # we have only past here, so no past-vs-future separator
                           end_date=end_date, major_tick_interval_days=14)
        self.format_x_axis(residuals_plot,
                           start_date=start_date,
                           now_date=None,
                           end_date=end_date, major_tick_interval_days=14)
        E_plot.legend()
        xlims = residuals_plot.get_xlim()
        residuals_plot.plot(xlims, [0, 0], c='black', linestyle='--')

        plt.savefig(
            self.forecast_data_interface.forecast_paths.get_residuals_plot(group_name)
        )
        plt.close(fig)
        print(f"Cases fit and beta residuals plot for {group} {group_name} is done")

    def create_final_draws_plot(self, group, compartments=('Cases', 'Deaths', 'R_effective'),
                                R_effective_in_log=True, linestyle="solid", transparency=0.1,
                                color=('orange', 'red', 'blue'), quantiles=(0.05, 0.95)):

        compartment_to_col = {
            'Cases': OUTPUT_DRAWS_CASES,
            'Deaths': OUTPUT_DRAWS_DEATHS,
            'R_effective': OUTPUT_DRAWS_REFF
        }

        group_name = self.id2loc[group]
        fig = plt.figure(figsize=(12, 3 * 6))
        grid = plt.GridSpec(3, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()

        for i, compartment in enumerate(compartments):

            compartment_data = self.data[group][compartment_to_col[compartment]]
            ax = fig.add_subplot(grid[i, 0])

            time = pd.to_datetime(compartment_data[self.col_date])
            start_date = time.to_list()[0]
            end_date = time.to_list()[-1]
            if compartment == 'Deaths':
                now_date = pd.to_datetime(
                    compartment_data[compartment_data[self.col_observed] == 1][self.col_date]
                ).to_list()[-1]
            else:
                now_date = None

            draw_num = 0
            draws = []

            while f"draw_{draw_num}" in compartment_data.columns:

                draw_name = f"draw_{draw_num}"
                draws.append(compartment_data[draw_name].to_numpy())

                if compartment == "R_effective":
                    if R_effective_in_log is True:
                        ax.semilogy(
                            time,
                            compartment_data[draw_name],
                            linestyle=linestyle, c=color[i],
                            alpha=transparency
                        )
                    else:
                        ax.plot(
                            time, compartment_data[draw_name], linestyle=linestyle,
                            c=color[i], alpha=transparency
                        )
                else:
                    ax.plot(
                        time, compartment_data[draw_name], linestyle=linestyle,
                        c=color[i], alpha=transparency
                    )
                draw_num += 1

            # plot mean and std
            mean_draw = np.nanmean(draws, axis=0)
            low_quantile = np.nanquantile(draws, quantiles[0], axis=0)
            high_quantile = np.nanquantile(draws, quantiles[1], axis=0)

            if compartment == "R_effective":
                ax.plot([start_date, end_date], [1, 1], linestyle='--', c="black")

            if compartment == "R_effective" and R_effective_in_log is True:
                ax.semilogy(time, mean_draw, linestyle="-", c="black", alpha=1, linewidth=1.5)
                ax.semilogy(time, low_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)
                ax.semilogy(time, high_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)

            else:
                ax.plot(time, mean_draw, linestyle="-", c="black", alpha=1, linewidth=1.5)
                ax.plot(time, low_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)
                ax.plot(time, high_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)

            self.format_x_axis(
                ax, start_date=start_date, now_date=now_date, end_date=end_date,
                major_tick_interval_days=14
            )
            ax.set_title(f"{group_name}: {compartment}")

        print(f"Final draws plot for {group} {group_name} is done")

        plt.savefig(
            self.forecast_data_interface.forecast_paths.get_final_draw_plots(group_name)
        )
        plt.close(fig)

    def plot_covariates(self, groups=None, covariates=None, base_plot_figsize=(12, 6), output_dir="."):
        raise NotImplementedError("this method is not currently implemented")
        # if covariates is None:
        #     covariates = self.covariates.keys()

        # for covariate in covariates:
        #     data_cov = self.covariates[covariate]
        #     if groups is None:
        #         groups = data_cov['location_id'].unique()

        #     fig = plt.figure(figsize=(base_plot_figsize[0], len(groups) * base_plot_figsize[1]))
        #     grid = plt.GridSpec(len(groups), 1, wspace=0.1, hspace=0.4)
        #     fig.autofmt_xdate()

        #     for i, group in enumerate(groups):
        #         group_name = self.id2loc[group]
        #         ax = fig.add_subplot(grid[i, 0])

        #         data = data_cov[data_cov['location_id'] == group]

        #         time = pd.to_datetime(data[self.col_date])
        #         values = data[covariate]
        #         nan_idx = values.isna()
        #         total_missing = sum(nan_idx)

        #         ax.scatter(time[~nan_idx], values[~nan_idx], c='b', label="data")
        #         ax.scatter(time[nan_idx], [0] * total_missing, c='r', label="NaNs")
        #         ax.legend()

        #         start_date = time.to_list()[0]
        #         end_date = time.to_list()[-1]
        #         now_date = pd.to_datetime(data[data[self.col_observed] == 1][self.col_date]).to_list()[-1]

        #         self.format_x_axis(ax, start_date, now_date, end_date, major_tick_interval_days=28)
        #         ax.set_ylabel(covariate)

        #         ax.set_title(f"{group_name}: {covariate} (missing points: {total_missing})")

        #     plt.savefig(os.path.join(output_dir, f"{covariate}_scatterplot.png"))
        #     print(f"Scatter plot for {covariate} is done")
        #     plt.close(fig)

    def plot_beta_fitting_process(self, df, group, cov_list, output_dir="."):
        fig = plt.figure(figsize=(12, 6))
        fig.autofmt_xdate()
        grid = plt.GridSpec(1, 1)
        ax_main = fig.add_subplot(grid[0, 0])

        time = pd.to_datetime(df[self.col_date])
        observed_idx = df[self.col_observed] == 1
        observed_time = pd.to_datetime(df[observed_idx][self.col_date])
        true_beta = df[observed_idx]["true_beta"].to_numpy()

        ax_main.scatter(observed_time, true_beta, label="true beta")

        prev_cov_names = []
        for i, covariate in enumerate(cov_list):
            values = df[covariate]
            prev_cov_names.append(covariate)
            ax_main.plot(time, values, label=" + ".join(prev_cov_names))

        ax_main.legend()

        start_date = time.to_list()[0]
        end_date = time.to_list()[-1]
        now_date = pd.to_datetime(df[df[self.col_observed] == 1][self.col_date]).to_list()[-1]

        self.format_x_axis(ax_main, start_date, now_date, end_date, major_tick_interval_days=14)

        group_name = self.id2loc[group]

        ax_main.set_title(group_name)
        ax_main.set_ylabel("Beta")
        plt.savefig(os.path.join(output_dir, f"{group_name}_beta_sequential_fit.png"))
        print(f"Sequential fit plot for {group_name} is done.")
        plt.close(fig)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("--location-id", type=int, required=True)
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

    visualizer = Visualizer(
        forecast_version=args.forecast_version,
        scenario=args.scenario,
        groups=[args.location_id],
        col_group=static_vars.INFECTION_COL_DICT['COL_LOC_ID'],
        col_date=static_vars.INFECTION_COL_DICT['COL_DATE']
    )
    visualizer.create_trajectories_plot(group=args.location_id)
    visualizer.create_final_draws_plot(group=args.location_id)
    visualizer.create_beta_fit_and_residuals_plot(group=args.location_id)


if __name__ == "__main__":
    main()

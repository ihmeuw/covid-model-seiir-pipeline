import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from seiir_model_pipeline.core.versioner import Directories
from seiir_model_pipeline.core.versioner import load_regression_settings, load_forecast_settings

ODE_BETA_FIT = "ode_beta_fit"
COEFFICIENTS_FIT = "coefficients_fit"
PARAMETERS_FIT = "param/usr/local/bin/lolcateters_fit"
ODE_COMPONENTS_FORECAST = "ode_forecast"
OUTPUT_DRAWS_CASES = "output_draws_cases"
OUTPUT_DRAWS_DEATHS = "output_draws_deaths"
OUTPUT_DRAWS_REFF = "output_draws_reff"


class Visualizer:

    def __init__(self, directories: Directories, groups: list = None, exclude_groups: list = None,
                 col_group="loc_id", col_date='date', col_observed='observed', covariates=()):

        self.directories = directories
        self.col_group = col_group
        self.col_date = col_date
        self.col_observed = col_observed
        self.groups = groups

        if exclude_groups is not None:
            for exclude_group in exclude_groups:
                self.groups.remove(exclude_group)

        self.data = {
            group: {
                ODE_BETA_FIT: [],
                COEFFICIENTS_FIT: [],
                ODE_COMPONENTS_FORECAST: [],
                OUTPUT_DRAWS_CASES: None,
                OUTPUT_DRAWS_DEATHS: None,
                OUTPUT_DRAWS_REFF: None
            } for group in self.groups
        }

        self.params_for_draws = []
        self.covariates = {}

        self.regression_settings = load_regression_settings(directories.regression_version)
        self.forecast_settings = load_forecast_settings(directories.forecast_version)

        self.location_metadata = pd.read_csv(
            self.directories.get_location_metadata_file(
                self.regression_settings.location_set_version_id)
        )
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # read beta regression draws
        for group in groups:
            path_to_regression_draws_for_group = os.path.join(directories.regression_beta_fit_dir, str(group))
            if os.path.isdir(path_to_regression_draws_for_group):
                for filename in os.listdir(directories.regression_beta_fit_dir):
                    if filename.startswith("fit_draw_") and filename.endswith(".csv"):
                        draw_df = pd.read_csv(os.path.join(path_to_regression_draws_for_group, filename))
                        # It's assumed that draw_df contains only the `group` group exclusively
                        self.data[group][ODE_BETA_FIT].append(draw_df)
                    else:
                        continue

        # read components forecast
        for group in groups:
            path_to_compartments_draws_for_group = os.path.join(directories.forecast_component_draw_dir, str(group))
            if os.path.isdir(path_to_compartments_draws_for_group):
                for filename in os.listdir(path_to_compartments_draws_for_group):
                    if filename.startswith("draw_") and filename.endswith(".csv"):
                        draw_df = pd.read_csv(os.path.join(path_to_compartments_draws_for_group, filename))
                        self.data[group][ODE_COMPONENTS_FORECAST].append(draw_df)
                    else:
                        continue
            else:
                error_msg = f"ODE Components forecast for the group with {col_group} = {group} is not found"
                print("Error: " + error_msg)

        #  read final draws
        if os.path.isdir(directories.forecast_output_draw_dir):
            for group in groups:
                self.data[group][OUTPUT_DRAWS_CASES] = pd.read_csv(
                    os.path.join(directories.forecast_output_draw_dir, f"cases_{group}.csv"))
                self.data[group][OUTPUT_DRAWS_DEATHS] = pd.read_csv(
                    os.path.join(directories.forecast_output_draw_dir, f"deaths_{group}.csv"))
                self.data[group][OUTPUT_DRAWS_REFF] = pd.read_csv(
                    os.path.join(directories.forecast_output_draw_dir, f"reff_{group}.csv"))

        for covariate in covariates:
            covariate_file = directories.get_covariate_file(
                covariate_name=covariate,
                covariate_version=self.forecast_settings.covariate_version
            )
            if os.path.exists(covariate_file):
                self.covariates[covariate] = pd.read_csv(covariate_file)
            else:
                raise ValueError(f"Can't find the file for covariate {covariate}: {covariate_file} does not exist.")

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

    def create_trajectories_plot(self, group, output_dir="plots",
                                 compartments=('S', 'E', 'I1', 'I2', 'R', 'beta'),
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

        plt.savefig(os.path.join(output_dir, f"trajectories_{group_name}.png"))
        plt.close(fig)

    def create_final_draws_plot(self, group, compartments=('Cases', 'Deaths', 'R_effective'),
                                R_effective_in_log=True, output_dir="plots",
                                linestyle="solid", transparency=0.1,
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
            now_date = pd.to_datetime(
                compartment_data[compartment_data[self.col_observed] == 1][self.col_date]
            ).to_list()[-1]

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

        plt.savefig(os.path.join(output_dir, f"final_draws_refflog_{group_name}.png"))
        plt.close(fig)

    def plot_covariates(self, groups=None, covariates=None, base_plot_figsize=(12, 6), output_dir="."):
        if covariates is None:
            covariates = self.covariates.keys()

        for covariate in covariates:
            data_cov = self.covariates[covariate]
            if groups is None:
                groups = data_cov['location_id'].unique()

            fig = plt.figure(figsize=(base_plot_figsize[0], len(groups) * base_plot_figsize[1]))
            grid = plt.GridSpec(len(groups), 1, wspace=0.1, hspace=0.4)
            fig.autofmt_xdate()

            for i, group in enumerate(groups):
                group_name = self.id2loc[group]
                ax = fig.add_subplot(grid[i, 0])

                data = data_cov[data_cov['location_id'] == group]

                time = pd.to_datetime(data[self.col_date])
                values = data[covariate]
                nan_idx = values.isna()
                total_missing = sum(nan_idx)

                ax.scatter(time[~nan_idx], values[~nan_idx], c='b', label="data")
                ax.scatter(time[nan_idx], [0] * total_missing, c='r', label="NaNs")
                ax.legend()

                start_date = time.to_list()[0]
                end_date = time.to_list()[-1]
                now_date = pd.to_datetime(data[data[self.col_observed] == 1][self.col_date]).to_list()[-1]

                self.format_x_axis(ax, start_date, now_date, end_date, major_tick_interval_days=28)
                ax.set_ylabel(covariate)

                ax.set_title(f"{group_name}: {covariate} (missing points: {total_missing})")

            plt.savefig(os.path.join(output_dir, f"{covariate}_scatterplot.png"))
            print(f"Scatter plot for {covariate} is done")
            plt.close(fig)

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


class PlotBetaCoef:
    def __init__(self, directories: Directories):

        self.directories = directories

        # load settings
        self.settings = load_regression_settings(directories.regression_version)

        self.path_to_location_metadata = self.directories.get_location_metadata_file(
            self.settings.location_set_version_id)

        self.path_to_coef_dir = self.directories.regression_coefficient_dir
        self.path_to_savefig = self.directories.regression_diagnostic_dir

        # load metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load coef
        df_coef = [
            pd.read_csv(self.directories.get_draw_coefficient_file(i))
            for i in range(self.settings.n_draws)
        ]

        # organize information
        self.covs = np.sort(list(self.settings.covariates.keys()))
        self.covs = np.append(self.covs, 'intercept')

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
            plt.figure(figsize=(8, 20))
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


class PlotBetaResidual:
    def __init__(self, directories: Directories):

        self.directories = directories

        # load settings
        self.settings = load_regression_settings(directories.regression_version)
        self.path_to_location_metadata = self.directories.get_location_metadata_file(
            self.settings.location_set_version_id)
        self.path_to_betas_dir = self.directories.regression_beta_fit_dir
        self.path_to_savefig = self.directories.regression_diagnostic_dir

        # load location metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load the beta data
        df_beta = [
            pd.read_csv(self.directories.get_draw_beta_fit_file(draw_id=i))[[
                'loc_id',
                'date',
                'days',
                'beta',
                'beta_pred'
            ]].dropna()
            for i in range(self.settings.n_draws)
        ]

        # location information
        self.loc_ids = np.sort(list(df_beta[0]['loc_id'].unique()))
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])
        self.num_locs = len(self.locs)

        # compute RMSE
        self.rmse_data = np.vstack([
            np.array([self.get_rmse(df, loc_id) for loc_id in self.loc_ids])
            for df in df_beta
        ])

    @staticmethod
    def get_rmse(df, loc_id):
        beta = df.loc[df.loc_id == loc_id, 'beta'].values
        pred_beta = df.loc[df.loc_id == loc_id, 'beta_pred'].values

        return np.sqrt(np.mean((beta - pred_beta) ** 2))

    def plot_residual(self):
        fig, ax = plt.subplots(self.num_locs, 1, figsize=(8, 4 * self.num_locs))
        for i, loc in enumerate(self.locs):
            ax[i].hist(self.rmse_data[:, i])
            ax[i].set_title(loc)
        plt.savefig(self.path_to_savefig / f'residual_rmse_histo.pdf',
                    bbox_inches='tight')

        plt.figure(figsize=(8, 20))
        sort_idx = np.argsort(self.rmse_data.mean(axis=0))
        plt.boxplot(self.rmse_data[:, sort_idx], vert=False, showfliers=False,
                    boxprops=dict(linewidth=0.5),
                    whiskerprops=dict(linewidth=0.5))
        plt.grid(b=True)
        plt.box(on=None)
        plt.yticks(ticks=np.arange(self.num_locs) + 1,
                   labels=self.locs[sort_idx])
        plt.title('Beta Regression Residual RMSE')
        plt.savefig(self.path_to_savefig / f'residual_rmse_boxplot.pdf',
                    bbox_inches='tight')

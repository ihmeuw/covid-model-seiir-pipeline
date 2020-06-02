from argparse import ArgumentParser
import logging

from covid_model_seiir_pipeline.diagnostics.visualizer import PlotBetaCoef
from covid_model_seiir_pipeline.core.versioner import args_to_directories

log = logging.getLogger(__name__)


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


def get_args():
    """
    Gets arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("--regression-version", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    log.info("Initiating SEIIR diagnostics.")

    # Load metadata
    args.forecast_version = None
    directories = args_to_directories(args)

    handle = PlotBetaCoef(directories)
    handle.plot_coef()

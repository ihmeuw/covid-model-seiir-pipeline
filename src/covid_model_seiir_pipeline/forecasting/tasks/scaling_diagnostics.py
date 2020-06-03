from argparse import ArgumentParser
import logging

from covid_model_seiir_pipeline.diagnostics.visualizer import PlotBetaScaling
from covid_model_seiir_pipeline.core.versioner import args_to_directories

log = logging.getLogger(__name__)


class PlotBetaScaling:
    def __init__(self, directories: Directories):

        self.directories = directories

        # load settings
        self.settings = load_regression_settings(directories.regression_version)
        self.path_to_location_metadata = self.directories.get_location_metadata_file(
            self.settings.location_set_version_id
        )
        self.path_to_beta_scaling = self.directories.forecast_beta_scaling_dir
        self.path_to_savefig = self.directories.forecast_diagnostic_dir

        # load location metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load locations
        self.loc_ids = np.array([
            file_name.split('_')[0]
            for file_name in os.listdir(self.path_to_beta_scaling)
        ]).astype(int)
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])

        # load data
        self.scales_data = np.hstack([
            pd.read_csv(self.directories.location_beta_scaling_file(loc_id))[
                'beta_scales'
            ].values[:, None]
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
        plt.savefig(self.path_to_savefig/f'beta_scalings_boxplot.pdf',
                    bbox_inches='tight')


def get_args():
    """
    Gets arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("--regression-version", type=str, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    log.info("Initiating SEIIR diagnostics.")

    # Load metadata
    directories = args_to_directories(args)

    handle = PlotBetaScaling(directories)
    handle.plot_scales()

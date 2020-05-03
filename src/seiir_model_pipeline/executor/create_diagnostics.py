from argparse import ArgumentParser
import logging

from seiir_model.visualizer.visualizer import PlotBetaCoef, PlotBetaResidual, Visualizer

from seiir_model_pipeline.core.utils import get_locations
from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import load_regression_settings
from seiir_model_pipeline.core.versioner import COVARIATE_COL_DICT, INFECTION_COL_DICT


log = logging.getLogger(__name__)


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
    settings = load_regression_settings(args.regression_version)

    handle = PlotBetaCoef(directories)
    handle.plot_coef()
    
    handle = PlotBetaResidual(directories)
    handle.plot_residual()

    location_ids = get_locations(
        directories, settings.location_set_version_id
    )

    visualizer = Visualizer(
        directories, groups=location_ids,
        col_group=INFECTION_COL_DICT['COL_LOC_ID'],
        col_date=INFECTION_COL_DICT['COL_DATE']
    )

    for loc in location_ids:
        print(f"Location ID {loc}")
        visualizer.create_trajectories_plot(
            group=loc,
            output_dir=directories.forecast_diagnostic_dir
        )
        visualizer.create_final_draws_plot(
            group=loc,
            output_dir=directories.forecast_diagnostic_dir
        )

from argparse import ArgumentParser
import logging

from seiir_model_pipeline.diagnostics.visualizer import Visualizer
from seiir_model_pipeline.core.versioner import args_to_directories
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT

log = logging.getLogger(__name__)


def get_args():
    """
    Gets arguments from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("--regression-version", type=str, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--location-id", type=int, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    log.info("Initiating SEIIR diagnostics.")

    # Load metadata
    directories = args_to_directories(args)

    visualizer = Visualizer(
        directories, groups=[args.location_id],
        col_group=INFECTION_COL_DICT['COL_LOC_ID'],
        col_date=INFECTION_COL_DICT['COL_DATE']
    )
    visualizer.create_trajectories_plot(
        group=args.location_id,
        output_dir=directories.forecast_diagnostic_dir
    )
    visualizer.create_final_draws_plot(
        group=args.location_id,
        output_dir=directories.forecast_diagnostic_dir
    )


if __name__ == "__main__":
    main()

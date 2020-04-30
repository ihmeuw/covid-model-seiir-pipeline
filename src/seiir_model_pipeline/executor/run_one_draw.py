from argparse import ArgumentParser
import logging
import numpy as np

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.core.file_master import args_to_directories
from seiir_model_pipeline.core.versioner import load_settings
from seiir_model_pipeline.core import data
from seiir_model_pipeline.core import file_master
from seiir_model_pipeline.core import utils

log = logging.getLogger(__name__)

LOCATION_IDS = [1]  # Where are we going to pull this list? Hierarchies?
# call to db_queries.get_location_metadata(location_set_version_id=...)


def get_args():
    """
    Gets arguments from the command line for only one run.
    """
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--output-version", type=str, required=True)
    parser.add_argument("--warm-start", action='store_true', required=False)

    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR modeling pipeline.")
    log.info(f"Running for draw {args.draw_id}.")
    log.info(f"This will be output version {args.output_version}.")
    if args.warm_start:
        log.info("Will resume from after beta regression.")

    directories = args_to_directories(args)

    # load settings
    settings = load_settings(args.output_version)

    # load necessary data
    location_data = data.load_all_location_data(directories,
                                           location_ids=LOCATION_IDS)
    peak_data = data.load_peaked_dates(
        file_master.PEAK_DATE_FILE,
        file_master.PEAK_DATE_COL_DICT['COL_LOC_ID'],
        file_master.PEAK_DATE_COL_DICT['COL_DATE']
    )

    # construct the model run
    np.seed(args.draw_id)
    mr = ModelRunner()
    if not args.warm_start:
        mr.fit_beta_ode(utils.process_ode_process_input(
            settings,
            location_data,
            peak_data,
        ))


if __name__ == '__main__':
    main()

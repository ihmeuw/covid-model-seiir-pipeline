from argparse import ArgumentParser
import logging

from seiir_model_pipeline.core.file_master import args_to_directories
from seiir_model_pipeline.core.data import load_all_location_data
from seiir_model.model_runner import ModelRunner

log = logging.getLogger(__name__)

LOCATION_IDS = [1]  # Where are we going to pull this list? Hierarchies?


def get_args():
    """
    Gets arguments from the command line for only one run.
    """
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--infection-version", type=str, required=True)
    parser.add_argument("--covariate-version", type=str, required=True)
    parser.add_argument("--output-version", type=str, required=True)
    parser.add_argument("--covariates", nargs="+", required=True)
    parser.add_argument("--warm-start", action='store_true', required=False)

    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR modeling pipeline.")
    log.info(f"Running for draw {args.draw_id}.")
    log.info(f"Getting infection version from {args.infection_version}.")
    log.info(f"Getting covariate data version from {args.covariate_version}.")
    log.info(f"This will be output version {args.output_version}.")
    log.info(f"Using {args.covariates.join(', ')} as regression covariates for beta.")
    if args.warm_start:
        log.info("Will resume from after beta regression.")

    directories = args_to_directories(args)
    location_data = load_all_location_data(directories, location_ids=LOCATION_IDS)
    mr = ModelRunner()


if __name__ == '__main__':
    main()

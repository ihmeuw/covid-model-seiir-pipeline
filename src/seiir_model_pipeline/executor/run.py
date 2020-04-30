from argparse import ArgumentParser
import logging

from seiir_model_pipeline.core.file_master import args_to_directories
from seiir_model_pipeline.core.workflow import SEIIRWorkFlow

log = logging.getLogger(__name__)


def get_args():
    """
    Get arguments from the command line for this whole fun.
    """
    parser = ArgumentParser()
    parser.add_argument("--n-draws", type=int, required=True)
    parser.add_argument("--infection-version", type=str, required=True)
    parser.add_argument("--covariate-version", type=str, required=True)
    parser.add_argument("--output-version", type=str, required=True)
    parser.add_argument("--covariates", nargs="+", required=True)
    parser.add_argument("--warm-start", action='store_true', required=False)

    return parser.parse_args()


def main():
    args = get_args()

    log.info("Initiating SEIIR modeling pipeline.")
    log.info(f"Running for {args.n_draws}.")
    log.info(f"Getting infection version from {args.infection_version}.")
    log.info(f"Getting covariate data version from {args.covariate_version}.")
    log.info(f"This will be output version {args.output_version}.")
    log.info(f"Using {args.covariates.join(', ')} as regression covariates for beta.")
    if args.warm_start:
        log.info("Will resume from after beta regression.")

    directories = args_to_directories(args)
    wf = SEIIRWorkFlow(directories=directories)
    wf.attach_tasks(n_draws=args.n_draws, covariates=args.covariates, warm_start=args.warm_start)
    wf.run()


if __name__ == '__main__':
    main()

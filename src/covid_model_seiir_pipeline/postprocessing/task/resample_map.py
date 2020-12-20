from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Optional

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.postprocessing.specification import PostprocessingSpecification, POSTPROCESSING_JOBS
from covid_model_seiir_pipeline.postprocessing.model import resampling, loaders


def run_resample_map(postprocessing_version: str) -> None:
    postprocessing_spec = PostprocessingSpecification.from_path(
        Path(postprocessing_version) / static_vars.POSTPROCESSING_SPECIFICATION_FILE
    )
    workflow_spec = postprocessing_spec.workflow.task_specifications[POSTPROCESSING_JOBS.resample]
    resampling_params = postprocessing_spec.resampling
    data_interface = PostprocessingDataInterface.from_specification(postprocessing_spec)
    deaths = loaders.load_deaths(resampling_params.reference_scenario,
                                 data_interface,
                                 workflow_spec.num_cores)
    deaths = pd.concat(deaths, axis=1)
    deaths['date'] = pd.to_datetime(deaths['date'])
    resampling_map = resampling.build_resampling_map(deaths, resampling_params)
    data_interface.save_resampling_map(resampling_map)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--postprocessing-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)  # Debug level

    args = parse_arguments()
    run_resample_map(postprocessing_version=args.postprocessing_version)


if __name__ == '__main__':
    main()

from pathlib import Path

import click

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.specification import (
    PreprocessingSpecification,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model import (
    MEASURES,
)

logger = cli_tools.task_performance_logger


def run_preprocess_measure(preprocessing_version: str, measure: str) -> None:
    logger.info(f'Starting preprocessing for measure {measure}.', context='setup')

    spec_file = Path(preprocessing_version) / static_vars.PREPROCESSING_SPECIFICATION_FILE
    specification = PreprocessingSpecification.from_path(spec_file)
    data_interface = PreprocessingDataInterface.from_specification(specification)

    try:
        MEASURES[measure](data_interface)
    except KeyError:
        raise ValueError(f'Unknown preprocessing measure {measure}.  Available measures: {list(MEASURES)}.')

    logger.report()
    

@click.command()
@cli_tools.with_task_preprocessing_version
@cli_tools.with_measure
@cli_tools.add_verbose_and_with_debugger
def preprocess_measure(preprocessing_version: str, measure: str,
                       verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_preprocess_measure, logger, with_debugger)
    run(preprocessing_version=preprocessing_version,
        measure=measure)


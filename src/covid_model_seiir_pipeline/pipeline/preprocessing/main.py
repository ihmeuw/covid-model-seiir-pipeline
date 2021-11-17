from typing import Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.pipeline.preprocessing.specification import PreprocessingSpecification
from covid_model_seiir_pipeline.pipeline.preprocessing.data import PreprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.preprocessing.workflow import PreprocessingWorkflow


def do_preprocessing(*args, **kwargs) -> PreprocessingSpecification:
    pass


def preprocessing_main(app_metadata: cli_tools.Metadata,
                       preprocessing_specification: PreprocessingSpecification,
                       preprocess_only: bool):
    pass


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
def preprocess(*args, **kwargs):
    pass

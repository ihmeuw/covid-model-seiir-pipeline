from typing import Optional, Union

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
@cli_tools.with_specification(PreprocessingSpecification)
@cli_tools.with_location_specification
@cli_tools.with_version(paths.MODEL_INPUTS_ROOT)
@cli_tools.with_version(paths.AGE_SPECIFIC_RATES_ROOT)
@cli_tools.with_version(paths.MORTALITY_SCALARS_ROOT)
@cli_tools.with_version(paths.MASK_USE_OUTPUT_ROOT)
@cli_tools.with_version(paths.MOBILITY_COVARIATES_OUTPUT_ROOT)
@cli_tools.with_version(paths.PNEUMONIA_OUTPUT_ROOT)
@cli_tools.with_version(paths.POPULATION_DENSITY_OUTPUT_ROOT)
@cli_tools.with_version(paths.TESTING_OUTPUT_ROOT)
@cli_tools.with_version(paths.VARIANT_OUTPUT_ROOT)
@cli_tools.with_version(paths.VACCINE_COVERAGE_OUTPUT_ROOT)
@cli_tools.with_version(paths.VACCINE_EFFICACY_ROOT)
@cli_tools.add_output_options(paths.SEIR_PREPROCESS_ROOT)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
def preprocess(run_metadata: cli_tools.RunMetadata,
               specification: str,
               location_specification: Union[str, int],
               output_root: str, mark_best: bool, production_tag: str,
               preprocess_only: bool,
               verbose: int, with_debugger: bool, **input_versions):
    pass

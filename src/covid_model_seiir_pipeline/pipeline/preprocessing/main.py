from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.pipeline.preprocessing.specification import PreprocessingSpecification
from covid_model_seiir_pipeline.pipeline.preprocessing.data import PreprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.preprocessing.workflow import PreprocessingWorkflow
from covid_model_seiir_pipeline.pipeline.preprocessing.model import (
    MEASURES,
)


def do_preprocessing(run_metadata: cli_tools.RunMetadata,
                     specification: PreprocessingSpecification,
                     location_specification: Optional[str],
                     output_root: Optional[str], mark_best: bool, production_tag: str,
                     preprocess_only: bool,
                     with_debugger: bool,
                     input_versions: Dict[str, cli_tools.VersionInfo]) -> PreprocessingSpecification:
    specification, run_metadata = cli_tools.resolve_version_info(specification, run_metadata, input_versions)

    locations_set_version_id, location_set_file = cli_tools.get_location_info(
        location_specification,
        specification.data.location_set_version_id,
        specification.data.location_set_file,
    )

    output_root = cli_tools.get_output_root(output_root, specification.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    specification.data.location_set_version_id = locations_set_version_id
    specification.data.location_set_file = location_set_file
    specification.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['preprocessing_specification'] = specification.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(preprocessing_main,
                                         logger, with_debugger)
    app_metadata, _ = main(specification, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)
    return specification


def preprocessing_main(app_metadata: cli_tools.Metadata,
                       preprocessing_specification: PreprocessingSpecification,
                       preprocess_only: bool):
    logger.info(f'Starting preprocessing for version {preprocessing_specification.data.output_root}.')

    # init high level objects
    data_interface = PreprocessingDataInterface.from_specification(preprocessing_specification)

    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_specification(preprocessing_specification)

    # Grab canonical location list from arguments
    hierarchy = data_interface.load_hierarchy_from_primary_source(
        location_set_version_id=preprocessing_specification.data.location_set_version_id,
        location_file=preprocessing_specification.data.location_set_file
    )
    # save location info
    data_interface.save_modeling_hierarchy(hierarchy)

    # build workflow and launch
    if not preprocess_only:
        workflow = PreprocessingWorkflow(preprocessing_specification.data.output_root,
                                         preprocessing_specification.workflow)
        workflow.attach_tasks(measures=MEASURES.keys(), scenarios=preprocessing_specification.data.vaccine_scenarios)
        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')
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
               specification: PreprocessingSpecification,
               location_specification: Optional[str],
               output_root: str, mark_best: bool, production_tag: str,
               preprocess_only: bool,
               verbose: int, with_debugger: bool,
               **input_versions: cli_tools.VersionInfo):
    cli_tools.configure_logging_to_terminal(verbose)
    do_preprocessing(
        run_metadata=run_metadata,
        specification=specification,
        location_specification=location_specification,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        preprocess_only=preprocess_only,
        with_debugger=with_debugger,
        input_versions=input_versions,
    )

    logger.info('**Done**')

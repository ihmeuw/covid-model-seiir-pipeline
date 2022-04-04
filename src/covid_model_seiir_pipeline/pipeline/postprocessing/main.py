from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools
from covid_model_seiir_pipeline.pipeline.postprocessing.specification import PostprocessingSpecification
from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.postprocessing.workflow import PostprocessingWorkflow
from covid_model_seiir_pipeline.pipeline.postprocessing import model


def do_postprocessing(run_metadata: cli_tools.RunMetadata,
                      specification: PostprocessingSpecification,
                      output_root: Optional[str], mark_best: bool, production_tag: str,
                      preprocess_only: bool,
                      with_debugger: bool,
                      input_versions: Dict[str, cli_tools.VersionInfo]) -> PostprocessingSpecification:
    specification, run_metadata = cli_tools.resolve_version_info(specification, run_metadata, input_versions)
    if specification.data.seir_counterfactual_version:
        specification.data.seir_forecast_version = ''

    output_root = cli_tools.get_output_root(output_root, specification.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    specification.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['postprocessing_specification'] = specification.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(postprocessing_main,
                                         logger, with_debugger)
    app_metadata, _ = main(specification, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)

    return specification


def postprocessing_main(app_metadata: cli_tools.Metadata,
                        specification: PostprocessingSpecification,
                        preprocess_only: bool):
    logger.info(f'Starting postprocessing for version {specification.data.output_root}.')

    data_interface = PostprocessingDataInterface.from_specification(specification)

    data_interface.make_dirs(scenario=specification.data.scenarios)
    data_interface.save_specification(specification)

    if not preprocess_only:
        workflow = PostprocessingWorkflow(specification.data.output_root,
                                          specification.workflow)
        known_covariates = list(model.COVARIATES)
        measures = [*model.MEASURES, *model.COMPOSITE_MEASURES,
                    *model.MISCELLANEOUS, *known_covariates]
        workflow.attach_tasks(measures, specification.data.scenarios)

        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')

    logger.info(f'Postprocessing version {specification.data.output_root} complete.')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(PostprocessingSpecification)
@cli_tools.add_output_options(paths.SEIR_FINAL_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_version(paths.SEIR_FORECAST_OUTPUTS)
@cli_tools.with_version(paths.SEIR_COUNTERFACTUAL_ROOT)
def postprocess(run_metadata,
                specification,
                output_root, mark_best, production_tag,
                preprocess_only,
                verbose, with_debugger,
                **input_versions):
    cli_tools.configure_logging_to_terminal(verbose)

    do_postprocessing(
        run_metadata=run_metadata,
        specification=specification,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        preprocess_only=preprocess_only,
        with_debugger=with_debugger,
        input_versions=input_versions,
    )

    logger.info('**Done**')

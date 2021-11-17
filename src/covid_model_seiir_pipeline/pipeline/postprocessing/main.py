from typing import Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools
from covid_model_seiir_pipeline.pipeline.postprocessing.specification import PostprocessingSpecification
from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.postprocessing.workflow import PostprocessingWorkflow
from covid_model_seiir_pipeline.pipeline.postprocessing import model


def do_postprocessing(run_metadata: cli_tools.RunMetadata,
                      postprocessing_specification: str,
                      forecast_version: Optional[str],
                      mortality_ratio_version: Optional[str],
                      preprocess_only: bool,
                      output_root: Optional[str], mark_best: bool, production_tag: str,
                      with_debugger: bool) -> PostprocessingSpecification:
    postprocessing_spec = PostprocessingSpecification.from_path(postprocessing_specification)

    input_versions = {
        'forecast_version': cli_tools.VersionInfo(
            forecast_version,
            postprocessing_spec.data.forecast_version,
            paths.SEIR_FORECAST_OUTPUTS,
            'forecast_metadata',
            True,
        ),
        'mortality_ratio_version': cli_tools.VersionInfo(
            mortality_ratio_version,
            postprocessing_spec.data.mortality_ratio_version,
            paths.MORTALITY_AGE_PATTERN_ROOT,
            'mortality_ratio_metadata',
            True,
        ),
    }

    postprocessing_spec, run_metadata = cli_tools.resolve_version_info(
        postprocessing_spec,
        run_metadata,
        input_versions,
    )

    output_root = cli_tools.get_output_root(output_root,
                                            postprocessing_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    postprocessing_spec.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['postprocessing_specification'] = postprocessing_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(postprocessing_main,
                                         logger, with_debugger)
    app_metadata, _ = main(postprocessing_spec, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)
    return postprocessing_spec


def postprocessing_main(app_metadata: cli_tools.Metadata,
                        postprocessing_specification: PostprocessingSpecification,
                        preprocess_only: bool):
    logger.info(f'Starting postprocessing for version {postprocessing_specification.data.output_root}.')

    data_interface = PostprocessingDataInterface.from_specification(postprocessing_specification)

    data_interface.make_dirs(scenario=postprocessing_specification.data.scenarios)
    data_interface.save_specification(postprocessing_specification)

    if not preprocess_only:
        workflow = PostprocessingWorkflow(postprocessing_specification.data.output_root,
                                          postprocessing_specification.workflow)
        known_covariates = list(model.COVARIATES)
        modeled_covariates = set(data_interface.get_covariate_names(postprocessing_specification.data.scenarios))
        unknown_covariates = modeled_covariates.difference(known_covariates + ['intercept'])
        if unknown_covariates:
            logger.warning("Some covariates that were modeled have no postprocessing configuration. "
                           "Postprocessing will produce no outputs for these covariates. "
                           f"Unknown covariates: {list(unknown_covariates)}")

        measures = [*model.MEASURES, *model.COMPOSITE_MEASURES,
                    *model.MISCELLANEOUS, *modeled_covariates.intersection(known_covariates)]
        workflow.attach_tasks(measures, postprocessing_specification.data.scenarios)

        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_postprocessing_specification
@cli_tools.with_forecast_version
@cli_tools.with_mortality_ratio_version
@cli_tools.add_preprocess_only
@cli_tools.add_output_options(paths.SEIR_FINAL_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def postprocess(run_metadata,
                postprocessing_specification,
                forecast_version,
                mortality_ratio_version,
                preprocess_only,
                output_root, mark_best, production_tag,
                verbose, with_debugger):
    cli_tools.configure_logging_to_terminal(verbose)

    do_postprocessing(
        run_metadata=run_metadata,
        postprocessing_specification=postprocessing_specification,
        forecast_version=forecast_version,
        mortality_ratio_version=mortality_ratio_version,
        preprocess_only=preprocess_only,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        with_debugger=with_debugger,
    )

    logger.info('**Done**')

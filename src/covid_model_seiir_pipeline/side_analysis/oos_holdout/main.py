from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import OOSHoldoutSpecification
from covid_model_seiir_pipeline.side_analysis.oos_holdout.data import OOSHoldoutDataInterface
from covid_model_seiir_pipeline.side_analysis.oos_holdout.workflow import OOSHoldoutWorkflow


def do_oos_holdout(run_metadata: cli_tools.RunMetadata,
                   specification: OOSHoldoutSpecification,
                   output_root: Optional[str], mark_best: bool, production_tag: str,
                   preprocess_only: bool,
                   with_debugger: bool,
                   input_versions: Dict[str, cli_tools.VersionInfo]) -> OOSHoldoutSpecification:
    specification, run_metadata = cli_tools.resolve_version_info(specification, run_metadata, input_versions)

    output_root = cli_tools.get_output_root(output_root, specification.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    specification.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['oos_holdout_specification'] = specification.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(oos_holdout_main,
                                         logger, with_debugger)
    app_metadata, _ = main(specification, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)

    return specification


def oos_holdout_main(app_metadata: cli_tools.Metadata,
                     specification: OOSHoldoutSpecification,
                     preprocess_only: bool):
    logger.info(f'Starting OOS holdout analysis for version {specification.data.output_root}.')

    # init high level objects
    data_interface = OOSHoldoutDataInterface.from_specification(specification)

    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_specification(specification)

    # build workflow and launch
    if not preprocess_only:
        workflow = OOSHoldoutWorkflow(specification.data.output_root,
                                           specification.workflow)
        workflow.attach_tasks(n_draws=data_interface.get_n_draws(), measures=[], plot_types=[])
        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')

    logger.info(f'OOS Holdout version {specification.data.output_root} complete.')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(OOSHoldoutSpecification)
@cli_tools.add_output_options(paths.SEIR_OOS_ANALYSIS_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_version(paths.SEIR_FORECAST_OUTPUTS)
def oos_holdout(run_metadata,
                specification,
                output_root, mark_best, production_tag,
                preprocess_only,
                verbose, with_debugger,
                **input_versions):
    cli_tools.configure_logging_to_terminal(verbose)

    do_oos_holdout(
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

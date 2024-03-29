from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(run_metadata: cli_tools.RunMetadata,
                       specification: RegressionSpecification,
                       output_root: Optional[str], mark_best: bool, production_tag: str,
                       preprocess_only: bool,
                       with_debugger: bool,
                       input_versions: Dict[str, cli_tools.VersionInfo]) -> RegressionSpecification:
    specification, run_metadata = cli_tools.resolve_version_info(specification, run_metadata, input_versions)

    output_root = cli_tools.get_output_root(output_root, specification.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    specification.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['regression_specification'] = specification.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(beta_regression_main,
                                         logger, with_debugger)
    app_metadata, _ = main(specification, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)

    return specification


def beta_regression_main(app_metadata: cli_tools.Metadata,
                         specification: RegressionSpecification,
                         preprocess_only: bool):
    logger.info(f'Starting beta regression for version {specification.data.output_root}.')

    # init high level objects
    data_interface = RegressionDataInterface.from_specification(specification)

    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_specification(specification)

    # Grab canonical location list from arguments
    hierarchy = data_interface.load_hierarchy('pred')
    # Filter to the intersection of what's available from the infection data.
    location_ids = data_interface.filter_location_ids(hierarchy)
    # save location info
    data_interface.save_location_ids(location_ids)

    # build workflow and launch
    if not preprocess_only:
        regression_wf = RegressionWorkflow(specification.data.output_root,
                                           specification.workflow)
        regression_wf.attach_tasks(n_draws=data_interface.get_n_draws())
        try:
            regression_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')

    logger.info(f'Regression version {specification.data.output_root} complete.')

@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(RegressionSpecification)
@cli_tools.add_output_options(paths.SEIR_REGRESSION_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_version(paths.SEIR_FIT_ROOT)
@cli_tools.with_version(paths.SEIR_COVARIATE_PRIORS_ROOT, allow_default=False)
@cli_tools.with_version(paths.SEIR_REGRESSION_OUTPUTS, allow_default=False, name='coefficient')
def regress(run_metadata,
            specification,
            output_root, mark_best, production_tag,
            preprocess_only,
            verbose, with_debugger,
            **input_versions):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    do_beta_regression(
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

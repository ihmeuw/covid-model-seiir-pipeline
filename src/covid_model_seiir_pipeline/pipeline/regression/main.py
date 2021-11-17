from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(run_metadata: cli_tools.RunMetadata,
                       regression_specification: str,
                       location_specification: Optional[str],
                       output_root: Optional[str], mark_best: bool, production_tag: str,
                       preprocess_only: bool,
                       with_debugger: bool,
                       **input_versions: Dict[str, cli_tools.VersionInfo]) -> RegressionSpecification:
    regression_spec = RegressionSpecification.from_path(regression_specification)

    regression_spec, run_metadata = cli_tools.resolve_version_info(regression_spec, run_metadata, input_versions)

    locations_set_version_id, location_set_file = cli_tools.get_location_info(
        location_specification,
        regression_spec.data.location_set_version_id,
        regression_spec.data.location_set_file
    )

    output_root = cli_tools.get_output_root(output_root, regression_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    regression_spec.data.location_set_version_id = locations_set_version_id
    regression_spec.data.location_set_file = location_set_file
    regression_spec.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['regression_specification'] = regression_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(beta_regression_main,
                                         logger, with_debugger)
    app_metadata, _ = main(regression_spec, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)

    return regression_spec


def beta_regression_main(app_metadata: cli_tools.Metadata,
                         regression_specification: RegressionSpecification,
                         preprocess_only: bool):
    logger.info(f'Starting beta regression for version {regression_specification.data.output_root}.')

    # init high level objects
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_specification(regression_specification)

    # Grab canonical location list from arguments
    hierarchy = data_interface.load_hierarchy_from_primary_source(
        location_set_version_id=regression_specification.data.location_set_version_id,
        location_file=regression_specification.data.location_set_file
    )
    # Filter to the intersection of what's available from the infection data.
    location_ids = data_interface.filter_location_ids(hierarchy)

    # Check to make sure we have all the covariates we need
    data_interface.check_covariates(regression_specification.covariates)

    # save location info
    data_interface.save_location_ids(location_ids)
    data_interface.save_hierarchy(hierarchy)

    # build workflow and launch
    if not preprocess_only:
        regression_wf = RegressionWorkflow(regression_specification.data.output_root,
                                           regression_specification.workflow)
        regression_wf.attach_tasks(n_draws=data_interface.get_n_draws())
        try:
            regression_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(RegressionSpecification)
@cli_tools.with_location_specification
@cli_tools.with_version(paths.PAST_INFECTIONS_ROOT)
@cli_tools.with_version(paths.SEIR_COVARIATES_OUTPUT_ROOT)
@cli_tools.with_version(paths.WANING_IMMUNITY_OUTPUT_ROOT)
@cli_tools.with_version(paths.SEIR_COVARIATE_PRIORS_ROOT, False)
@cli_tools.with_version(paths.SEIR_REGRESSION_OUTPUTS, False, 'coefficient')
@cli_tools.add_output_options(paths.SEIR_REGRESSION_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
def regress(run_metadata,
            regression_specification,
            location_specification,
            infection_version, covariates_version, waning_version,
            priors_version, coefficient_version,
            output_root, mark_best, production_tag,
            preprocess_only,
            verbose, with_debugger):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    do_beta_regression(
        run_metadata=run_metadata,
        regression_specification=regression_specification,
        location_specification=location_specification,
        infection_version=infection_version,
        covariates_version=covariates_version,
        waning_version=waning_version,
        priors_version=priors_version,
        coefficient_version=coefficient_version,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        preprocess_only=preprocess_only,
        with_debugger=with_debugger,
    )

    logger.info('**Done**')

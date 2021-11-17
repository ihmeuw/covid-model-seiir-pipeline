from typing import Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools

from covid_model_seiir_pipeline.side_analysis.parameter_fit.specification import FitSpecification
from covid_model_seiir_pipeline.side_analysis.parameter_fit.workflow import FitWorkflow
from covid_model_seiir_pipeline.side_analysis.parameter_fit.data import FitDataInterface


def do_parameter_fit(run_metadata: cli_tools.RunMetadata,
                     fit_specification: str,
                     infection_version: Optional[str],
                     covariates_version: Optional[str],
                     coefficient_version: Optional[str],
                     variant_version: Optional[str],
                     location_specification: Optional[str],
                     preprocess_only: bool,
                     output_root: Optional[str], mark_best: bool, production_tag: str,
                     with_debugger: bool) -> FitSpecification:
    fit_spec = FitSpecification.from_path(fit_specification)

    input_versions = {
        'infection_version': cli_tools.VersionInfo(
            infection_version,
            fit_spec.data.infection_version,
            paths.PAST_INFECTIONS_ROOT,
            'infections_metadata',
            True,
        ),
        'covariate_version': cli_tools.VersionInfo(
            covariates_version,
            fit_spec.data.covariate_version,
            paths.SEIR_COVARIATES_OUTPUT_ROOT,
            'covariates_metadata',
            True,
        ),
        'coefficient_version': cli_tools.VersionInfo(
            coefficient_version,
            fit_spec.data.coefficient_version,
            paths.SEIR_REGRESSION_OUTPUTS,
            'coefficient_metadata',
            False,
        ),
        'variant_version': cli_tools.VersionInfo(
            variant_version,
            fit_spec.data.variant_version,
            paths.VARIANT_OUTPUT_ROOT,
            'variant_metadata',
            True,
        ),
    }
    fit_spec, run_metadata = cli_tools.resolve_version_info(fit_spec, run_metadata, input_versions)

    locations_set_version_id, location_set_file = cli_tools.get_location_info(
        location_specification,
        fit_spec.data.location_set_version_id,
        fit_spec.data.location_set_file
    )

    output_root = cli_tools.get_output_root(None, fit_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    fit_spec.data.location_set_version_id = locations_set_version_id
    fit_spec.data.location_set_file = location_set_file
    fit_spec.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['regression_specification'] = fit_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(parameter_fit_main,
                                         logger, with_debugger)
    app_metadata, _ = main(fit_spec, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)

    return fit_spec


def parameter_fit_main(app_metadata: cli_tools.Metadata,
                       fit_specification: FitSpecification,
                       preprocess_only: bool):
    logger.info(f'Starting beta fit for version {fit_specification.data.output_root}.')

    data_interface = FitDataInterface.from_specification(fit_specification)

    # Grab canonical location list from arguments
    hierarchy = data_interface.load_hierarchy_from_primary_source(
        location_set_version_id=fit_specification.data.location_set_version_id,
        location_file=fit_specification.data.location_set_file
    )
    location_ids = data_interface.filter_location_ids(hierarchy)

    data_interface.make_dirs(scenario=list(fit_specification.scenarios))
    data_interface.save_specification(fit_specification)
    data_interface.save_location_ids(location_ids)
    data_interface.save_hierarchy(hierarchy)

    if not preprocess_only:
        fit_wf = FitWorkflow(fit_specification.data.output_root,
                             fit_specification.workflow)
        n_draws = data_interface.get_n_draws()

        fit_wf.attach_tasks(n_draws=n_draws,
                            scenarios=fit_specification.scenarios)
        try:
            fit_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_fit_specification
@cli_tools.with_infection_version
@cli_tools.with_covariates_version
@cli_tools.with_coefficient_version
@cli_tools.with_variant_version
@cli_tools.with_location_specification
@cli_tools.add_preprocess_only
@cli_tools.add_output_options(paths.SEIR_REGRESSION_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def parameter_fit(run_metadata,
                  fit_specification,
                  infection_version, covariates_version, coefficient_version, variant_version,
                  location_specification,
                  preprocess_only,
                  output_root, mark_best, production_tag,
                  verbose, with_debugger):
    cli_tools.configure_logging_to_terminal(verbose)

    do_parameter_fit(
        run_metadata=run_metadata,
        fit_specification=fit_specification,
        infection_version=infection_version,
        covariates_version=covariates_version,
        coefficient_version=coefficient_version,
        variant_version=variant_version,
        location_specification=location_specification,
        preprocess_only=preprocess_only,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        with_debugger=with_debugger,
    )

    logger.info('**Done**')

from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools
from covid_model_seiir_pipeline.pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.pipeline.forecasting.workflow import ForecastWorkflow


def do_forecast(run_metadata: cli_tools.RunMetadata,
                specification: ForecastSpecification,
                output_root: Optional[str], mark_best: bool, production_tag: str,
                preprocess_only: bool,
                with_debugger: bool,
                input_versions: Dict[str, cli_tools.VersionInfo]) -> ForecastSpecification:
    specification, run_metadata = cli_tools.resolve_version_info(specification, run_metadata, input_versions)

    output_root = cli_tools.get_output_root(output_root, specification.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    specification.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['forecast_specification'] = specification.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(forecast_main,
                                         logger, with_debugger)
    app_metadata, _ = main(specification, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)

    return specification


def forecast_main(app_metadata: cli_tools.Metadata,
                  specification: ForecastSpecification,
                  preprocess_only: bool):
    logger.info(f'Starting beta forecast for version {specification.data.output_root}.')

    data_interface = ForecastDataInterface.from_specification(specification)

    # Check scenario covariates the same as regression covariates and that
    # covariate data versions match.
    # data_interface.check_covariates(specification.scenarios)

    data_interface.make_dirs(scenario=list(specification.scenarios))
    data_interface.save_specification(specification)

    if not preprocess_only:
        forecast_wf = ForecastWorkflow(specification.data.output_root,
                                       specification.workflow)
        n_draws = data_interface.get_n_draws()

        forecast_wf.attach_tasks(n_draws=n_draws,
                                 scenarios=specification.scenarios)
        try:
            forecast_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')

    logger.info(f'Forecast version {specification.data.output_root} complete.')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(ForecastSpecification)
@cli_tools.add_output_options(paths.SEIR_FORECAST_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_version(paths.SEIR_REGRESSION_OUTPUTS)
def forecast(run_metadata,
             specification,
             output_root, mark_best, production_tag,
             preprocess_only,
             verbose, with_debugger,
             **input_versions):
    """Perform beta forecast for a set of scenarios on a regression."""
    cli_tools.configure_logging_to_terminal(verbose)

    do_forecast(
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

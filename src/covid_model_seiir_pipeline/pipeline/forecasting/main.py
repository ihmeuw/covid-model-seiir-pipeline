from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import cli_tools
from covid_model_seiir_pipeline.pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.pipeline.forecasting.workflow import ForecastWorkflow


def do_forecast(run_metadata: cli_tools.RunMetadata,
                forecast_specification: str,
                output_root: Optional[str], mark_best: bool, production_tag: str,
                preprocess_only: bool,
                with_debugger: bool,
                **input_versions: Dict[str, cli_tools.VersionInfo]) -> ForecastSpecification:
    forecast_spec = ForecastSpecification.from_path(forecast_specification)

    forecast_spec, run_metadata = cli_tools.resolve_version_info(forecast_spec, run_metadata, input_versions)

    output_root = cli_tools.get_output_root(output_root,
                                            forecast_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    forecast_spec.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['forecast_specification'] = forecast_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(forecast_main,
                                         logger, with_debugger)
    app_metadata, _ = main(forecast_spec, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)
    return forecast_spec


def forecast_main(app_metadata: cli_tools.Metadata,
                  forecast_specification: ForecastSpecification,
                  preprocess_only: bool):
    logger.info(f'Starting beta forecast for version {forecast_specification.data.output_root}.')

    data_interface = ForecastDataInterface.from_specification(forecast_specification)

    # Check scenario covariates the same as regression covariates and that
    # covariate data versions match.
    data_interface.check_covariates(forecast_specification.scenarios)

    data_interface.make_dirs(scenario=list(forecast_specification.scenarios))
    data_interface.save_specification(forecast_specification)

    if not preprocess_only:
        forecast_wf = ForecastWorkflow(forecast_specification.data.output_root,
                                       forecast_specification.workflow)
        n_draws = data_interface.get_n_draws()

        forecast_wf.attach_tasks(n_draws=n_draws,
                                 scenarios=forecast_specification.scenarios)
        try:
            forecast_wf.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(ForecastSpecification)
@cli_tools.with_version(paths.SEIR_REGRESSION_OUTPUTS)
@cli_tools.with_version(paths.SEIR_COVARIATES_OUTPUT_ROOT)
@cli_tools.add_output_options(paths.SEIR_FORECAST_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
def forecast(run_metadata,
             forecast_specification,
             regression_version,
             covariates_version,
             output_root, mark_best, production_tag,
             preprocess_only,
             verbose, with_debugger):
    """Perform beta forecast for a set of scenarios on a regression."""
    cli_tools.configure_logging_to_terminal(verbose)

    do_forecast(
        run_metadata=run_metadata,
        forecast_specification=forecast_specification,
        regression_version=regression_version,
        covariates_version=covariates_version,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        preprocess_only=preprocess_only,
        with_debugger=with_debugger,
    )

    logger.info('**Done**')

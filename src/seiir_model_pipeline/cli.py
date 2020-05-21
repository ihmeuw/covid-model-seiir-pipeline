from pathlib import Path

import click
from covid_shared import cli_tools, paths
from loguru import logger

from seiir_model_pipeline import regression, ode_fit, forecasting, utilities


@click.group()
def seiir():
    pass


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('fit_specification')
@click.option('--infection-version',
              type=click.Path(file_okay=False),
              help="Which version of infectionator inputs to use in the"
                   "regression.")
@click.option('-o', '--output-root',
              type=click.Path(file_okay=False),
              show_default=True)
@click.option('-b', '--mark-best', 'mark_dir_as_best',
              is_flag=True,
              help='Marks the new outputs as best in addition to marking '
                   'them as latest.')
@click.option('-p', '--production-tag',
              type=click.STRING,
              help='Tags this run as a production run.')
@cli_tools.add_verbose_and_with_debugger
def fit(run_metadata,
        fit_specification,
        infection_version,
        output_root, mark_dir_as_best, production_tag,
        verbose, with_debugger):
    """Runs a beta fit on a set of infection data."""
    cli_tools.configure_logging_to_terminal(verbose)

    fit_spec = ode_fit.FitSpecification.from_path(fit_specification)

    infection_version = utilities.get_version(infection_version, fit_spec.data.infection_version)
    infection_root = cli_tools.get_last_stage_directory(
        infection_version, last_stage_root=paths.INFECTIONATOR_OUTPUTS
    )
    fit_spec.data.infection_version = str(infection_root.resolve())

    run_metadata.update_from_path('infectionator_metadata',
                                  infection_root / paths.METADATA_FILE_NAME)

    output_root = utilities.get_output_root(output_root, fit_spec.data.output_root, paths.SEIR_FIT_OUTPUTS)
    fit_spec.data.output_root = str(output_root)

    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)
    run_metadata['ode_fit_specification'] = fit_spec.to_dict()
    fit_spec.dump(run_directory / 'fit_specification.yaml')

    main = cli_tools.monitor_application(ode_fit.do_beta_fit,
                                         logger, with_debugger)
    app_metadata, _ = main(fit_spec, infection_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_dir_as_best, production_tag)

    logger.info('**Done**')


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('regression_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--ode-fit-version',
              type=click.Path(file_okay=False),
              help="Which version of ode fit inputs to use in the"
                   "regression.")
@click.option('--covariates-version',
              type=click.Path(file_okay=False),
              help=('Which version of the covariates to use in the '
                    'regression.'))
@click.option('-o', '--output-root',
              type=click.Path(file_okay=False),
              show_default=True)
@click.option('-b', '--mark-best', 'mark_dir_as_best',
              is_flag=True,
              help='Marks the new outputs as best in addition to marking '
                   'them as latest.')
@click.option('-p', '--production-tag',
              type=click.STRING,
              help='Tags this run as a production run.')
@cli_tools.add_verbose_and_with_debugger
def regress(run_metadata,
            regression_specification,
            ode_fit_version, covariates_version,
            output_root, mark_dir_as_best, production_tag,
            verbose, with_debugger):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    regression_spec = regression.RegressionSpecification(regression_specification)

    ode_fit_version = utilities.get_version(ode_fit_version, regression_spec.data.ode_fit_version)
    ode_fit_root = cli_tools.get_last_stage_directory(
        ode_fit_version, last_stage_root=paths.SEIR_FIT_OUTPUTS
    )
    regression_spec.data.infection_version = str(ode_fit_root.resolve())

    covariates_version = utilities.get_version(covariates_version, regression_spec.data.covariate_version)
    covariates_root = cli_tools.get_last_stage_directory(
        covariates_version, last_stage_root=paths.SEIR_COVARIATES_OUTPUT_ROOT
    )
    regression_spec.data.covariate_version = str(covariates_root.resolve())

    for key, input_root in zip(['ode_fit_metadata', 'covariates_metadata'],
                               [ode_fit_root, covariates_root]):
        run_metadata.update_from_path(key, input_root / paths.METADATA_FILE_NAME)

    output_root = utilities.get_output_root(output_root, regression_spec.data.output_root,
                                            paths.SEIR_REGRESSION_OUTPUTS)
    regression_spec.data.output_root = str(output_root)

    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)
    run_metadata['regression_specification'] = regression_spec.to_dict()
    regression_spec.dump(run_directory / 'regression_specification.yaml')

    main = cli_tools.monitor_application(regression.do_beta_regression,
                                         logger, with_debugger)
    app_metadata, _ = main(regression_spec, ode_fit_root,
                           covariates_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_dir_as_best, production_tag)

    logger.info('**Done**')


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('forecast_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--regression-version',
              type=click.Path(file_okay=False),
              help="Which version of ode fit inputs to use in the"
                   "regression.")
@click.option('-o', '--output-root',
              type=click.Path(file_okay=False),
              show_default=True)
@click.option('-b', '--mark-best', 'mark_dir_as_best',
              is_flag=True,
              help='Marks the new outputs as best in addition to marking '
                   'them as latest.')
@click.option('-p', '--production-tag',
              type=click.STRING,
              help='Tags this run as a production run.')
@cli_tools.add_verbose_and_with_debugger
def forecast(run_metadata,
             forecast_specification,
             regression_version,
             output_root, mark_dir_as_best, production_tag,
             verbose, with_debugger):
    """Perform beta forecast for a set of scenarios on a regression."""
    cli_tools.configure_logging_to_terminal(verbose)

    forecast_spec = forecasting.ForecastSpecification.from_path(forecast_specification)

    regression_version = utilities.get_version(regression_version, forecast_spec.data.regression_version)
    regression_root = cli_tools.get_last_stage_directory(
        regression_version, last_stage_root=paths.SEIR_REGRESSION_OUTPUTS
    )
    forecast_spec.data.regression_version = str(regression_root.resolve())

    run_metadata.update_from_path('regression_metadata', regression_root / paths.METADATA_FILE_NAME)

    output_root = utilities.get_output_root(output_root, forecast_spec.data.output_root,
                                            paths.SEIR_FORECAST_OUTPUTS)
    forecast_spec.data.output_root = str(output_root)

    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)
    run_metadata['forecast_specification'] = forecast_spec.to_dict()
    forecast_spec.dump(run_directory / 'forecast_specification.yaml')

    main = cli_tools.monitor_application(forecasting.do_beta_forecast,
                                         logger, with_debugger)
    app_metadata, _ = main(forecast_spec, regression_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_dir_as_best, production_tag)

    logger.info('**Done**')

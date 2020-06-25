import click
from covid_shared import cli_tools, paths
from loguru import logger

from covid_model_seiir_pipeline import regression, ode_fit, forecasting, utilities


@click.group()
def seiir():
    pass


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('fit_specification')
@click.option('--infection-version',
              type=click.Path(file_okay=False),
              default=paths.BEST_LINK,
              help="Which version of infectionator inputs to use in the"
                   "regression.")
@click.option('-l', '--location-specification',
              type=click.STRING,
              help="Either a location set version id used to pull a list of"
                   "locations to run, or a full path to a file describing"
                   "the location set.")
@cli_tools.add_output_options(paths.SEIR_FIT_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def fit(run_metadata,
        fit_specification,
        infection_version,
        location_specification,
        output_root, mark_best, production_tag,
        verbose, with_debugger):
    """Runs a beta fit on a set of infection data."""
    cli_tools.configure_logging_to_terminal(verbose)

    fit_spec = ode_fit.FitSpecification.from_path(fit_specification)

    # Resolve CLI overrides and specification values with defaults into
    # final run arguments.
    infection_root = utilities.get_input_root(infection_version,
                                              fit_spec.data.infection_version,
                                              paths.INFECTIONATOR_OUTPUTS)
    locations_set_version_id, location_set_file = utilities.get_location_metadata(
        location_specification,
        fit_spec.data.location_set_version_id,
        fit_spec.data.location_set_file
    )
    output_root = utilities.get_output_root(output_root, fit_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    # Make the fit specification consistent with the resolved arguments
    # and dump to disk.
    fit_spec.data.infection_version = str(infection_root)
    fit_spec.data.location_set_version_id = locations_set_version_id
    fit_spec.data.location_set_file = location_set_file
    fit_spec.data.output_root = str(run_directory)
    fit_spec.dump(run_directory / 'fit_specification.yaml')

    # Update the run metadata with our extra info.
    run_metadata.update_from_path('infectionator_metadata',
                                  infection_root / paths.METADATA_FILE_NAME)
    run_metadata['output_path'] = str(run_directory)
    run_metadata['ode_fit_specification'] = fit_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    main = cli_tools.monitor_application(ode_fit.do_beta_fit,
                                         logger, with_debugger)
    app_metadata, _ = main(fit_spec, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

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
@cli_tools.add_output_options(paths.SEIR_REGRESSION_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def regress(run_metadata,
            regression_specification,
            ode_fit_version, covariates_version,
            output_root, mark_best, production_tag,
            verbose, with_debugger):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    regression_spec = regression.RegressionSpecification.from_path(regression_specification)

    # Resolve CLI overrides and specification values with defaults into
    # final run arguments.
    ode_fit_root = utilities.get_input_root(ode_fit_version,
                                            regression_spec.data.ode_fit_version,
                                            paths.SEIR_FIT_OUTPUTS)
    covariates_root = utilities.get_input_root(covariates_version,
                                               regression_spec.data.covariate_version,
                                               paths.SEIR_COVARIATES_OUTPUT_ROOT)
    output_root = utilities.get_output_root(output_root,
                                            regression_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    # Make the regression specification consistent with the resolved arguments
    # and dump to disk.
    regression_spec.data.ode_fit_version = str(ode_fit_root)
    regression_spec.data.covariate_version = str(covariates_root)
    regression_spec.data.output_root = str(run_directory)
    regression_spec.dump(run_directory / 'regression_specification.yaml')

    for key, input_root in zip(['ode_fit_metadata', 'covariates_metadata'],
                               [ode_fit_root, covariates_root]):
        run_metadata.update_from_path(key, input_root / paths.METADATA_FILE_NAME)
    run_metadata['output_path'] = str(run_directory)
    run_metadata['regression_specification'] = regression_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    main = cli_tools.monitor_application(regression.do_beta_regression,
                                         logger, with_debugger)
    app_metadata, _ = main(regression_spec, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

    logger.info('**Done**')


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('forecast_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--regression-version',
              type=click.Path(file_okay=False),
              help="Which version of ode fit inputs to use in the"
                   "regression.")
@cli_tools.add_output_options(paths.SEIR_FORECAST_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def forecast(run_metadata,
             forecast_specification,
             regression_version,
             output_root, mark_best, production_tag,
             verbose, with_debugger):
    """Perform beta forecast for a set of scenarios on a regression."""
    cli_tools.configure_logging_to_terminal(verbose)

    forecast_spec = forecasting.ForecastSpecification.from_path(forecast_specification)

    regression_root = utilities.get_input_root(regression_version,
                                               forecast_spec.data.regression_version,
                                               paths.SEIR_REGRESSION_OUTPUTS)
    output_root = utilities.get_output_root(output_root,
                                            forecast_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    forecast_spec.data.regression_version = str(regression_root.resolve())
    forecast_spec.data.output_root = str(run_directory)
    forecast_spec.dump(run_directory / 'forecast_specification.yaml')

    run_metadata.update_from_path('regression_metadata', regression_root / paths.METADATA_FILE_NAME)
    run_metadata['output_path'] = str(run_directory)
    run_metadata['forecast_specification'] = forecast_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    main = cli_tools.monitor_application(forecasting.do_beta_forecast,
                                         logger, with_debugger)
    app_metadata, _ = main(forecast_spec, regression_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

    logger.info('**Done**')

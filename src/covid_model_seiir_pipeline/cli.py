from pathlib import Path

import click
from covid_shared import cli_tools, paths, shell_tools
from loguru import logger

from covid_model_seiir_pipeline import utilities
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.main import do_beta_regression
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification, PostprocessingSpecification
from covid_model_seiir_pipeline.forecasting.main import do_beta_forecast, do_postprocessing
from covid_model_seiir_pipeline.predictive_validity.specification import PredictiveValiditySpecification
from covid_model_seiir_pipeline.predictive_validity.main import do_predictive_validity


@click.group()
def seiir():
    """Top level entry point for running SEIIR pipeline stages."""
    pass


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('regression_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--infection-version',
              type=click.Path(file_okay=False),
              help="Which version of infectionator inputs to use in the"
                   "regression.")
@click.option('--covariates-version',
              type=click.Path(file_okay=False),
              help=('Which version of the covariates to use in the '
                    'regression.'))
@click.option('-l', '--location-specification',
              type=click.STRING,
              help="Either a location set version id used to pull a list of"
                   "locations to run, or a full path to a file describing"
                   "the location set.")
@click.option('--preprocess-only',
              is_flag=True,
              help="Only make the directory and set up the metadata. "
                   "Useful for setting up output directories for testing "
                   "tasks individually.")
@cli_tools.add_output_options(paths.SEIR_REGRESSION_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def regress(run_metadata,
            regression_specification,
            infection_version, covariates_version,
            location_specification,
            preprocess_only,
            output_root, mark_best, production_tag,
            verbose, with_debugger):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    regression_spec = RegressionSpecification.from_path(regression_specification)

    # Resolve CLI overrides and specification values with defaults into
    # final run arguments.
    infection_root = utilities.get_input_root(infection_version,
                                              regression_spec.data.infection_version,
                                              paths.INFECTIONATOR_OUTPUTS)
    covariates_root = utilities.get_input_root(covariates_version,
                                               regression_spec.data.covariate_version,
                                               paths.SEIR_COVARIATES_OUTPUT_ROOT)
    locations_set_version_id, location_set_file = utilities.get_location_metadata(
        location_specification,
        regression_spec.data.location_set_version_id,
        regression_spec.data.location_set_file
    )
    output_root = utilities.get_output_root(output_root, regression_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    # Make the regression specification consistent with the resolved arguments
    # and dump to disk.
    regression_spec.data.infection_version = str(infection_root)
    regression_spec.data.covariate_version = str(covariates_root)
    regression_spec.data.location_set_version_id = locations_set_version_id
    regression_spec.data.location_set_file = location_set_file
    regression_spec.data.output_root = str(run_directory)

    for key, input_root in zip(['infectionator_metadata', 'covariates_metadata'],
                               [infection_root, covariates_root]):
        run_metadata.update_from_path(key, input_root / paths.METADATA_FILE_NAME)
    run_metadata['output_path'] = str(run_directory)
    run_metadata['regression_specification'] = regression_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    main = cli_tools.monitor_application(do_beta_regression,
                                         logger, with_debugger)
    app_metadata, _ = main(regression_spec, preprocess_only)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

    logger.info('**Done**')


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('forecast_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--postprocesssing-specification',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--regression-version',
              type=click.Path(file_okay=False),
              help="Which version of ode fit inputs to use in the"
                   "regression.")
@click.option('--covariates-version',
              type=click.Path(file_okay=False),
              help=('Which version of the covariates to use in the '
                    'regression.'))
@click.option('--preprocess-only',
              is_flag=True,
              help="Only make the directory and set up the metadata. "
                   "Useful for setting up output directories for testing "
                   "tasks individually.")
@cli_tools.add_output_options(paths.SEIR_FORECAST_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
@click.pass_context
def forecast(run_metadata,
             forecast_specification,
             postprocessing_specification,
             regression_version,
             covariates_version,
             preprocess_only,
             output_root, mark_best, production_tag,
             verbose, with_debugger,
             ctx):
    """Perform beta forecast for a set of scenarios on a regression."""
    cli_tools.configure_logging_to_terminal(verbose)

    forecast_spec = ForecastSpecification.from_path(forecast_specification)

    regression_root = utilities.get_input_root(regression_version,
                                               forecast_spec.data.regression_version,
                                               paths.SEIR_REGRESSION_OUTPUTS)
    covariates_root = utilities.get_input_root(covariates_version,
                                               forecast_spec.data.covariate_version,
                                               paths.SEIR_COVARIATES_OUTPUT_ROOT)
    output_root = utilities.get_output_root(output_root,
                                            forecast_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    forecast_spec.data.regression_version = str(regression_root.resolve())
    forecast_spec.data.covariate_version = str(covariates_root.resolve())
    forecast_spec.data.output_root = str(run_directory)

    run_metadata.update_from_path('regression_metadata', regression_root / paths.METADATA_FILE_NAME)
    run_metadata['output_path'] = str(run_directory)
    run_metadata['forecast_specification'] = forecast_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    main = cli_tools.monitor_application(do_beta_forecast,
                                         logger, with_debugger)
    app_metadata, _ = main(forecast_spec, preprocess_only)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

    logger.info('**Done**')

    if postprocessing_specification is not None:
        logger.info('Starting postprocessing.')
        ctx.invoke(postprocess,
                   postprocessing_specification=postprocessing_specification,
                   forecast_version=forecast_spec.data.output_root,
                   mark_best=mark_best,
                   production_tag=production_tag,
                   verbose=verbose,
                   with_debugger=with_debugger)


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('postprocessing_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--forecast-version',
              type=click.Path(file_okay=False),
              help="Which version of ode fit inputs to use in the"
                   "regression.")
@cli_tools.add_output_options(paths.SEIR_FINAL_OUTPUTS)
@cli_tools.add_verbose_and_with_debugger
def postprocess(run_metadata,
                postprocessing_specification,
                forecast_version,
                output_root, mark_best, production_tag,
                verbose, with_debugger):
    cli_tools.configure_logging_to_terminal(verbose)

    postprocessing_spec = PostprocessingSpecification.from_path(postprocessing_specification)

    forecast_root = utilities.get_input_root(forecast_version,
                                             postprocessing_spec.data.forecast_version,
                                             paths.SEIR_FORECAST_OUTPUTS)
    output_root = utilities.get_output_root(output_root,
                                            postprocessing_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    postprocessing_spec.data.forecast_version = str(forecast_root.resolve())
    postprocessing_spec.data.output_root = str(run_directory)

    run_metadata.update_from_path('forecast_metadata', forecast_root / paths.METADATA_FILE_NAME)
    run_metadata['output_path'] = str(run_directory)
    run_metadata['postprocessing_specification'] = postprocessing_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    main = cli_tools.monitor_application(do_postprocessing,
                                         logger, with_debugger)
    app_metadata, _ = main(postprocessing_spec)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

    logger.info('**Done**')


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('regression_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('forecast_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.argument('predictive_validity_specification',
                type=click.Path(exists=True, dir_okay=False))
@cli_tools.add_verbose_and_with_debugger
def predictive_validity(run_metadata,
                        regression_specification,
                        forecast_specification,
                        predictive_validity_specification,
                        verbose, with_debugger):
    """Perform OOS predictive validity testing."""
    cli_tools.configure_logging_to_terminal(verbose)

    regression_spec = RegressionSpecification.from_path(regression_specification)
    forecast_spec = ForecastSpecification.from_path(forecast_specification)
    predictive_validity_spec = PredictiveValiditySpecification.from_path(predictive_validity_specification)

    run_directory = Path(predictive_validity_spec.output_root)
    shell_tools.mkdir(run_directory, exists_ok=True)

    main = cli_tools.monitor_application(do_predictive_validity, logger, with_debugger)
    app_metadata, _ = main(regression_spec, forecast_spec, predictive_validity_spec)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    logger.info('Done')

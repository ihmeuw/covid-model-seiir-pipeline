import pkgutil

import click
from covid_shared import paths
from loguru import logger

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline import (
    pipeline,
    side_analysis,
)


@click.group()
def seiir():
    """Top level entry point for running SEIIR pipeline stages."""
    pass


# Loops over every pipeline stage and adds pipeline main to `seiir` command.
for package in [pipeline, side_analysis]:
    for importer, modname, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            m = __import__(modname)
            seiir.add_command(modname.COMMAND)


@seiir.command(name='run_all')
@cli_tools.pass_run_metadata()
@cli_tools.with_regression_specification
@cli_tools.with_forecast_specification
@cli_tools.with_postprocessing_specification
@cli_tools.with_diagnostics_specification
@cli_tools.with_mark_best
@cli_tools.with_production_tag
@cli_tools.add_verbose_and_with_debugger
def run_all(run_metadata,
            regression_specification, forecast_specification,
            postprocessing_specification, diagnostics_specification,
            mark_best, production_tag,
            verbose, with_debugger):
    """Run all stages of the SEIIR pipeline.

    This application is expected to be run in production only and provides
    no user-level interface for specifying output paths. The `data` block
    of the regression specification can specify a specific path for infection
    data and covariate data, but all output paths and downstream input
    paths will be inferred automatically.
    """
    base_metadata = run_metadata.to_dict()
    del base_metadata['start_time']
    #####################
    # Do the regression #
    #####################
    # Build our own run metadata since the injected version is shared across the
    # three pipeline stages.
    regression_run_metadata = cli_tools.RunMetadata()
    regression_run_metadata.update(base_metadata)

    cli_tools.configure_logging_to_terminal(verbose)

    regression_spec = pipeline.regression.APPLICATION_MAIN(
        run_metadata=regression_run_metadata,
        regression_specification=regression_specification,
        infection_version=None,
        covariates_version=None,
        priors_version=None,
        coefficient_version=None,
        location_specification=None,
        preprocess_only=False,
        output_root=paths.SEIR_REGRESSION_OUTPUTS,
        mark_best=mark_best,
        production_tag=production_tag,
        with_debugger=with_debugger,
    )

    logger.info('Regression finished. Starting forecast.')

    ###################
    # Do the forecast #
    ###################

    forecast_run_metadata = cli_tools.RunMetadata()
    forecast_run_metadata.update(base_metadata)

    # Get rid of last stage file handlers.
    logger.remove(2)
    logger.remove(3)

    forecast_spec = pipeline.forecasting.APPLICATION_MAIN(
        run_metadata=forecast_run_metadata,
        forecast_specification=forecast_specification,
        regression_version=regression_spec.data.output_root,
        covariates_version=regression_spec.data.covariate_version,
        preprocess_only=False,
        output_root=paths.SEIR_FORECAST_OUTPUTS,
        mark_best=mark_best,
        production_tag=production_tag,
        with_debugger=with_debugger,
    )

    logger.info('Forecast finished. Starting postprocessing.')

    #########################
    # Do the postprocessing #
    #########################

    postprocessing_run_metadata = cli_tools.RunMetadata()
    postprocessing_run_metadata.update(base_metadata)

    # Get rid of last stage file handlers so we get clean postprocessing logs.
    logger.remove(4)
    logger.remove(5)

    postprocessing_spec = pipeline.postprocessing.APPLICATION_MAIN(
        run_metadata=postprocessing_run_metadata,
        postprocessing_specification=postprocessing_specification,
        forecast_version=forecast_spec.data.output_root,
        mortality_ratio_version=None,
        preprocess_only=False,
        output_root=paths.SEIR_FINAL_OUTPUTS,
        mark_best=mark_best,
        production_tag=production_tag,
        with_debugger=with_debugger,
    )

    logger.info('Postprocessing finished. Starting diagnostics.')

    ######################
    # Do the diagnostics #
    ######################

    diagnostics_run_metadata = cli_tools.RunMetadata()
    diagnostics_run_metadata.update(base_metadata)

    # Get rid of last stage file handlers so we get clean diagnostics logs.
    logger.remove(6)
    logger.remove(7)

    diagnostics_spec = pipeline.diagnostics.APPLICATION_MAIN(
        run_metadata=diagnostics_run_metadata,
        diagnostics_specification=diagnostics_specification,
        preprocess_only=False,
        output_root=paths.SEIR_DIAGNOSTICS_OUTPUTS,
        mark_best=mark_best,
        production_tag=production_tag,
        with_debugger=with_debugger
    )

    logger.info('All stages complete!')

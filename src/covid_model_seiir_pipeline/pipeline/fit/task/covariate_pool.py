import click

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_covariate_pool(fit_version: str) -> None:
    logger.info('Starting beta fit.', context='setup')
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    n_samples = data_interface.get_n_total_draws()

    logger.info('Loading covariate data', context='read')
    # ... load covariates and first pass ifr data if needed here ...

    logger.info('Identifying best covariate combinations and inflection points.', context='model')
    covariate_options = model.make_covariate_pool(n_samples)

    logger.info('Writing covariate options', context='write')
    data_interface.save_covariate_options(covariate_options)

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.add_verbose_and_with_debugger
def covariate_pool(fit_version: str,
                   verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_covariate_pool, logger, with_debugger)
    run(fit_version=fit_version)

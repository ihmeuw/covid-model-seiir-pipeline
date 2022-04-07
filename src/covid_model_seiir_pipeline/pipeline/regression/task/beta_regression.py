import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
)
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.pipeline.regression import model


logger = cli_tools.task_performance_logger


def run_beta_regression(regression_version: str, draw_id: int) -> None:
    logger.info('Starting beta regression.', context='setup')
    # Build helper abstractions
    regression_specification = RegressionSpecification.from_version_root(regression_version)
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    logger.info('Loading regression input data', context='read')
    hierarchy = data_interface.load_hierarchy('pred')
    beta_fit = data_interface.load_fit_beta(draw_id, columns=['beta_all_infection'])
    beta_fit = beta_fit.loc[:, 'beta_all_infection'].rename('beta')
    # FIXME: Beta should be nan or positive here.
    beta_fit = beta_fit.loc[beta_fit > 0]
    covariates = data_interface.load_covariates(list(regression_specification.covariates))
    gaussian_priors = data_interface.load_priors(regression_specification.covariates.values())
    prior_coefficients = data_interface.load_prior_run_coefficients(draw_id=draw_id)
    if gaussian_priors and prior_coefficients:
        raise NotImplementedError

    logger.info('Fitting beta regression', context='compute_regression')
    coefficients = model.run_beta_regression(
        beta_fit,
        covariates,
        regression_specification.covariates.values(),
        gaussian_priors,
        prior_coefficients,
        hierarchy,
    )
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_hat')

    # Format and save data.
    logger.info('Prepping outputs', context='transform')
    betas = pd.concat([beta_fit, beta_hat], axis=1)

    logger.info('Writing outputs', context='write')
    data_interface.save_regression_beta(betas, draw_id=draw_id)
    data_interface.save_coefficients(coefficients, draw_id=draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_regression_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
def beta_regression(regression_version: str, draw_id: int,
                    verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_regression, logger, with_debugger)
    run(regression_version=regression_version,
        draw_id=draw_id)

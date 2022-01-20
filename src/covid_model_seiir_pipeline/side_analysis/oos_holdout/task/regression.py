import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.data import OOSHoldoutDataInterface
from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import OOSHoldoutSpecification
from covid_model_seiir_pipeline.pipeline.regression import model


logger = cli_tools.task_performance_logger


def run_oos_holdout_regression(oos_holdout_version: str, draw_id: int) -> None:
    logger.info('Starting oos regression.', context='setup')
    # Build helper abstractions
    specification = OOSHoldoutSpecification.from_version_root(oos_holdout_version)
    data_interface = OOSHoldoutDataInterface.from_specification(specification)

    logger.info('Loading regression input data', context='read')
    hierarchy = data_interface.load_hierarchy('pred')
    beta_fit = data_interface.load_fit_beta(draw_id, columns=['beta', 'round'])
    beta_fit = beta_fit.loc[(beta_fit['round'] == 2) & (beta_fit['beta'] > 0), 'beta']
    covariates = data_interface.load_covariates()
    gaussian_priors = data_interface.load_priors()
    prior_coefficients = data_interface.load_prior_run_coefficients(draw_id=draw_id)
    if gaussian_priors and prior_coefficients:
        raise NotImplementedError

    logger.info('Fitting beta regression', context='compute_regression')
    holdout_days = specification.parameters.holdout_weeks * 7
    beta_fit = beta_fit.drop(beta_fit.groupby('location_id').tail(holdout_days).index)
    if specification.parameters.run_regression:
        regression_spec = data_interface.forecast_data_interface.regression_data_interface.load_specification()
        coefficients = model.run_beta_regression(
            beta_fit,
            covariates,
            regression_spec.covariates.values(),
            gaussian_priors,
            prior_coefficients,
            hierarchy,
        )
    else:
        coefficients = prior_coefficients

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
@cli_tools.with_task_oos_holdout_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
def oos_holdout_regression(oos_holdout_version: str, draw_id: int,
                           verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_oos_holdout_regression, logger, with_debugger)
    run(oos_holdout_version=oos_holdout_version,
        draw_id=draw_id)

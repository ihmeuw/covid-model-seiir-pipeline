from pathlib import Path

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.pipeline.regression import model


logger = cli_tools.task_performance_logger


def run_beta_regression(regression_version: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta regression.', context='setup')
    # Build helper abstractions
    regression_spec_file = Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    regression_specification = RegressionSpecification.from_path(regression_spec_file)
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    logger.info('Loading ODE fit input data', context='read')
    past_infection_data = data_interface.load_past_infection_data(draw_id=draw_id)
    population = data_interface.load_total_population()
    vaccinations = data_interface.load_vaccine_info('reference')

    logger.info('Prepping ODE fit parameters.', context='transform')
    infections = model.clean_infection_data_measure(past_infection_data, 'infections')
    deaths = model.clean_infection_data_measure(past_infection_data, 'deaths')
    regression_params = regression_specification.regression_parameters.to_dict()
    ode_parameters = model.prepare_ode_fit_parameters(
        infections.index,
        population,
        vaccinations,
        regression_params,
        draw_id,
    )

    logger.info('Loading regression input data', context='read')
    covariates = data_interface.load_covariates(regression_specification.covariates)
    if regression_specification.data.coefficient_version:
        prior_coefficients = data_interface.load_prior_run_coefficients(draw_id=draw_id)
    else:
        prior_coefficients = None

    logger.info('Running ODE fit', context='compute_ode')
    beta_fit = model.run_ode_fit(
        infections=infections,
        ode_parameters=ode_parameters,
        progress_bar=progress_bar,
    )
    logger.info('Prepping regression.', context='transform')
    mr_data = model.align_beta_with_covariates(covariates, beta_fit, list(regression_specification.covariates))
    regressor = model.build_regressor(regression_specification.covariates.values(), prior_coefficients)
    logger.info('Fitting beta regression', context='compute_regression')
    coefficients = regressor.fit(mr_data, regression_specification.regression_parameters.sequential_refit)
    log_beta_hat = math.compute_beta_hat(covariates.reset_index(), coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    # Format and save data.
    logger.info('Prepping outputs', context='transform')
    merge_cols = ['location_id', 'date']
    # These two datasets are aligned, but go out into the future
    regression_betas = beta_hat.merge(covariates, on=merge_cols)
    # Regression betas include the forecast.  Subset to just the modeled past
    # based on data in beta fit.
    regression_betas = beta_fit.merge(regression_betas, how='left').sort_values(['location_id', 'date'])
    # There is more observed data than there is modeled, based on the day_shift
    # parameter and infection drops in the ode fit. Expand to the size of the
    # data, leaving NAs.
    merged = past_infections.reset_index().merge(regression_betas, how='left').sort_values(['location_id', 'date'])
    data_df = merged[['location_id', 'date', 'infections', 'deaths']].set_index(['location_id', 'date'])
    regression_betas = merged[regression_betas.columns].set_index(['location_id', 'date'])
    coefficients = coefficients.set_index('location_id')
    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    ode_params = ode_params.to_dict()
    draw_beta_params = pd.DataFrame({
        'params': ode_params.keys(),
        'values': ode_params.values(),
    })

    logger.info('Writing outputs', context='write')
    data_interface.save_infection_data(data_df, draw_id)
    data_interface.save_regression_betas(regression_betas, draw_id)
    data_interface.save_regression_coefficients(coefficients, draw_id)
    data_interface.save_beta_param_file(draw_beta_params, draw_id)
    data_interface.save_date_file(beta_start_end_dates, draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_regression_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_regression(regression_version: str, draw_id: int,
                    progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_regression, logger, with_debugger)
    run(regression_version=regression_version,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    beta_regression()

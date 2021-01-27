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


def run_beta_regression(regression_version: str, draw_id: int) -> None:
    logger.info('Starting beta regression.', context='setup')
    # Build helper abstractions
    regression_spec_file = Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    regression_specification = RegressionSpecification.from_path(regression_spec_file)
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    logger.info('Loading input data', context='read')
    location_ids = data_interface.load_location_ids()
    population = data_interface.load_five_year_population(location_ids).groupby('location_id')['population'].sum()

    past_infections = data_interface.load_past_infection_data(draw_id=draw_id, location_ids=location_ids)

    covariates = data_interface.load_covariates(regression_specification.covariates, location_ids)

    if regression_specification.data.coefficient_version:
        prior_coefficients = data_interface.load_prior_run_coefficients(draw_id=draw_id)
    else:
        prior_coefficients = None

    logger.info('Prepping ODE fit', context='transform')
    regression_params = regression_specification.regression_parameters.to_dict()
    ode_params = model.sample_parameters(draw_id, regression_params)

    logger.info('Running ODE fit', context='compute_ode')
    beta_fit = model.run_beta_fit(
        past_infections=past_infections['infections'].dropna(),  # Drop days with deaths but no infecs.
        population=population,
        location_ids=location_ids,
        ode_parameters=ode_params,
    )
    beta_start_end_dates = (beta_fit
                            .groupby('location_id')
                            .agg(start_date=('date', 'min'), end_date=('date', 'max'))
                            .reset_index())

    logger.info('Prepping regression.', context='transform')
    mr_data = model.align_beta_with_covariates(covariates, beta_fit, list(regression_specification.covariates))
    regressor = model.build_regressor(regression_specification.covariates.values(), prior_coefficients)
    logger.info('Fitting beta regression', context='compute_regression')
    coefficients = regressor.fit(mr_data, regression_specification.regression_parameters.sequential_refit)
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
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
    data_df = merged[['location_id', 'date', 'infections', 'deaths']]
    regression_betas = merged[regression_betas.columns]
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
def beta_regression(regression_version: str, draw_id: int,
                    verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_regression, logger, with_debugger)
    run(regression_version=regression_version,
        draw_id=draw_id)


if __name__ == '__main__':
    beta_regression()

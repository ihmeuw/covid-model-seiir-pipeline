from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Optional

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.math import compute_beta_hat
from covid_model_seiir_pipeline.static_vars import REGRESSION_SPECIFICATION_FILE
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression import model


def run_beta_regression(draw_id: int, regression_version: str) -> None:
    # Build helper abstractions
    logger.info('Beginning regression.')
    regression_spec_file = Path(regression_version) / REGRESSION_SPECIFICATION_FILE
    regression_specification = RegressionSpecification.from_path(regression_spec_file)
    data_interface = RegressionDataInterface.from_specification(regression_specification)
    # Load data
    logger.info('Loading regression data.')
    location_ids = data_interface.load_location_ids()
    location_data = data_interface.load_infection_data(draw_id, location_ids)
    location_data = {location_id: location_data[location_data['location_id'] == location_id].copy()
                     for location_id in location_ids}
    covariates = data_interface.load_covariates(regression_specification.covariates, location_ids)

    logger.info('Preparing inputs.')
    np.random.seed(draw_id)
    beta_fit_inputs = model.ODEProcessInput(
        df_dict=location_data,
        col_date='date',
        col_infections='infections_draw',
        col_pop='population',
        col_loc_id='location_id',
        col_lag_days='duration',
        col_observed='observed_deaths',
        alpha=regression_specification.parameters.alpha,
        sigma=regression_specification.parameters.sigma,
        gamma1=regression_specification.parameters.gamma1,
        gamma2=regression_specification.parameters.gamma2,
        solver_dt=regression_specification.parameters.solver_dt,
        day_shift=regression_specification.parameters.day_shift,
    )

    logger.info('Modeling past compartments data from infections.')
    ode_model = model.ODEProcess(beta_fit_inputs)
    beta_fit = ode_model.process()
    beta_fit['date'] = pd.to_datetime(beta_fit['date'])

    # Run regression
    logger.info('Running regression.')
    mr_data = model.align_beta_with_covariates(covariates, beta_fit, list(regression_specification.covariates))
    regressor = model.build_regressor(regression_specification.covariates.values())
    coefficients = regressor.fit(mr_data)
    log_beta_hat = compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    # Format and save data.
    logger.info('Formatting and writing outputs.')
    data_df = pd.concat(location_data.values())
    regression_betas = beta_hat.merge(covariates, on=['location_id', 'date'])
    regression_betas = beta_fit.merge(regression_betas, on=['location_id', 'date'], how='left')
    merged = (data_df
              .merge(regression_betas, on=['location_id', 'date'], how='outer')
              .sort_values(['location_id', 'date']))
    merged = merged[(merged['observed_deaths'] == 1) | (merged['deaths_draw'] > 0)]
    data_df = merged[data_df.columns]
    data_interface.save_location_data(data_df, draw_id)

    regression_betas = merged[regression_betas.columns]
    data_interface.save_regression_betas(regression_betas, draw_id)
    data_interface.save_regression_coefficients(coefficients, draw_id)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    draw_beta_params = ode_model.create_params_df()
    data_interface.save_beta_param_file(draw_beta_params, draw_id)

    beta_start_end_dates = ode_model.create_start_end_date_df()
    data_interface.save_date_file(beta_start_end_dates, draw_id)

    logger.info('**Done**')


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)  # Debug level
    args = parse_arguments()
    run_beta_regression(draw_id=args.draw_id, regression_version=args.regression_version)


if __name__ == '__main__':
    main()

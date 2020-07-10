from argparse import ArgumentParser, Namespace
from typing import Optional, Dict
import logging
from pathlib import Path
import shlex

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.math import compute_beta_hat
from covid_model_seiir_pipeline.forecasting import model
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface


log = logging.getLogger(__name__)


def run_beta_forecast(draw_id: int, forecast_version: str, scenario_name: str):
    log.info("Initiating SEIIR beta forecasting.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    location_ids = data_interface.load_location_ids()
    if isinstance(scenario_spec.theta, str):
        thetas = pd.read_csv(scenario_spec.theta).set_index('location_id')
    else:
        thetas = pd.DataFrame({'theta': scenario_spec.theta},
                              index=pd.Index(location_ids, name='location_id'))

    # Contains the start and end date for the data that was fit.
    # End dates vary based on how many days of leading indicator data
    # we use from the deaths model.
    dates_df = data_interface.load_dates_df(draw_id)
    # we just want a map between location id and day we transition to
    # prediction.
    transition_date = dates_df.set_index('location_id').sort_index()['end_date'].rename('date')

    # We'll use the beta and SEIR compartments from this data set to get
    # the ODE initial condition.
    beta_regression_df = data_interface.load_beta_regression(draw_id)

    # We'll use the covariates and coefficients to compute beta hat in the
    # future.
    covariates = data_interface.load_covariates(scenario_spec, location_ids)
    coefficients = data_interface.load_regression_coefficients(draw_id)

    # We'll use the same params in the ODE forecast as we did in the fit.
    beta_params = data_interface.load_beta_params(draw_id=draw_id)

    # Modeling starts

    # Align date in data sets
    # We want the past out of the regression data. Keep the overlap day.
    beta_regression_df = beta_regression_df.set_index('location_id').sort_index()
    idx = beta_regression_df.index
    beta_past = beta_regression_df.loc[beta_regression_df['date'] <= transition_date.loc[idx]].reset_index()

    # For covariates, we want the future.  Also keep the overlap day
    covariates = covariates.set_index('location_id').sort_index()
    idx = covariates.index
    covariate_pred = covariates.loc[covariates['date'] >= transition_date.loc[idx]].reset_index()

    log_beta_hat = compute_beta_hat(covariate_pred, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    betas = beta_hat.beta_pred.values
    days = beta_hat.date.values
    days = pd.to_datetime(days)
    times = np.array((days - days.min()).days)

    # FIXME: vectorize over location
    betas, scale_params = model.beta_shift(beta_past, betas, draw_id, **scenario_spec.beta_scaling)

    transition_day = beta_regression_df['date'] == transition_date.loc[beta_regression_df.index]
    compartments = ['S', 'E', 'I1', 'I2', 'R']
    initial_condition = beta_regression_df.loc[transition_day, compartments]

    forecasts = []

    for location_id in location_ids:
        log.info(f"On location id {location_id}")
        init_cond = initial_condition.loc[location_id].values
        total_population = initial_condition.sum()

        model_specs = model.SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=total_population,
        )

        ode_runner = model.ODERunner(model_specs, init_cond)

        loc_times = times.loc[location_id].values
        loc_betas = betas.loc[location_id].values
        loc_thetas = np.repeat(thetas.at[location_id], loc_betas)

        forecasted_components = ode_runner.get_solution(loc_times, loc_betas, loc_thetas)
        forecasted_components['date'] = days

        forecasts.append(forecasted_components)

    forecasts = pd.concat(forecasts)
    data_interface.save_forecasts(forecasts, scenario_name, draw_id)
    data_interface.save_beta_scales(scale_params, scenario_name, draw_id)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    run_beta_forecast(draw_id=args.draw_id,
                      forecast_version=args.forecast_version,
                      scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()

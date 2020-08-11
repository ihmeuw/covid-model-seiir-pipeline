from argparse import ArgumentParser, Namespace
from typing import Optional
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
        thetas = pd.read_csv(scenario_spec.theta).set_index('location_id')['theta']
    else:
        thetas = pd.Series({'theta': scenario_spec.theta},
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

    # Covariates and coefficients, and scaling parameters are
    # used to compute beta hat in the future.
    covariates = data_interface.load_covariates(scenario_spec, location_ids)
    coefficients = data_interface.load_regression_coefficients(draw_id)
    beta_scales = data_interface.load_beta_scales(scenario=scenario_name, draw_id=draw_id)

    # We'll use the same params in the ODE forecast as we did in the fit.
    beta_params = data_interface.load_beta_params(draw_id=draw_id)

    # Modeling starts

    # Grab the projection of the covariates into the future, keeping the
    # day of transition from past model to future model.
    covariates = covariates.set_index('location_id').sort_index()
    the_future = covariates['date'] >= transition_date.loc[covariates.index]
    covariate_pred = covariates.loc[the_future].reset_index()

    log_beta_hat = compute_beta_hat(covariate_pred, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    # Rescale the predictions of beta based on the residuals from the
    # regression.
    betas = model.beta_shift(beta_hat, beta_scales).set_index('location_id')

    transition_day = beta_regression_df['date'] == transition_date.loc[beta_regression_df.index]
    compartments = ['S', 'E', 'I1', 'I2', 'R']
    initial_condition = (beta_regression_df
                         .set_index('location_id')
                         .sort_index()
                         .loc[transition_day, compartments])

    forecasts = []
    for location_id in location_ids:
        log.info(f"On location id {location_id}")
        init_cond = initial_condition.loc[location_id].values
        total_population = init_cond.sum()

        model_specs = model.SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=total_population,
        )
        ode_runner = model.ODERunner(model_specs, init_cond)

        loc_betas = betas.loc[location_id].sort_values('date')
        loc_days = loc_betas['date']
        loc_times = np.array((loc_days - loc_days.min()).dt.days)
        loc_betas = loc_betas['beta_pred'].values
        loc_thetas = np.repeat(thetas.get(location_id, default=0), loc_betas.size)

        forecasted_components = ode_runner.get_solution(loc_times, loc_betas, loc_thetas)
        forecasted_components['date'] = loc_days.values
        forecasted_components['location_id'] = location_id
        forecasts.append(forecasted_components)

    forecasts = pd.concat(forecasts)
    # Concat past with future
    shared_columns = ['date', 'S', 'E', 'I1', 'I2', 'R', 'beta']
    components = (pd.concat([beta_regression_df.loc[~transition_day, shared_columns].reset_index(),
                             forecasts[['location_id'] + shared_columns]])
                    .sort_values(['location_id', 'date'])
                    .set_index(['location_id']))
    components['theta'] = thetas.reindex(components.index).fillna(0)
    data_interface.save_components(components, scenario_name, draw_id)

    infection_data = data_interface.load_infection_data(draw_id)

    observed_infections, modeled_infections = model.compute_infections(infection_data, components)
    infections = observed_infections.combine_first(modeled_infections)['cases_draw'].rename('infections')

    observed_deaths, modeled_deaths = model.compute_deaths(infection_data, modeled_infections)
    deaths = observed_deaths.combine_first(modeled_deaths).rename(columns={'deaths_draw': 'deaths'})

    r_effective = model.compute_effective_r(infection_data, components, beta_params)

    outputs = pd.concat([infections, deaths, r_effective], axis=1).dropna().reset_index()
    data_interface.save_outputs(outputs, scenario_name, draw_id)


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

from argparse import ArgumentParser, Namespace
from typing import Optional, Dict
import logging
from pathlib import Path
import shlex

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.model_runner import compute_beta_hat
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.model import get_ode_init_cond, SeiirModelSpecs


log = logging.getLogger(__name__)


def run_beta_forecast(draw_id: int, forecast_version: str, scenario_name: str):
    log.info("Initiating SEIIR beta forecasting.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]

    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    location_ids = data_interface.load_location_ids()
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

    ### Modeling starts

    # Align date in data sets
    # We want the past out of the regression data. Keep the overlap day.
    beta_regression_df = beta_regression_df.set_index('location_id').sort_index()
    beta_past = beta_regression_df.loc[beta_regression_df['date'] <= transition_date].reset_index()

    # For covariates, we want the future.  Also keep the overlap day
    covariates = covariates.set_index('location_id').sort_index()
    covariate_pred = covariates.loc[covariates['date'] >= transition_date].reset_index()

    log_beta_hat = compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    betas = beta_hat.beta_pred.values
    days = beta_hat.date.values
    days = pd.to_datetime(days)
    times = np.array((days - days.min()).days)

        # Anchor the betas at the last observed beta (fitted)
        # and scale everything into the future from this anchor value
        anchor_beta = beta_fit.beta[beta_fit.date == CURRENT_DATE].iloc[0]
        scale = anchor_beta / betas[0]
        scales.append(scale)
        # scale = scale + (1 - scale)/20.0*np.arange(betas.size)
        # scale[21:] = 1.0
        betas = betas * scale

        # Get initial conditions based on the beta fit for forecasting into the future
        init_cond = get_ode_init_cond(
            beta_ode_fit=beta_fit,
            current_date=CURRENT_DATE,
            location_id=location_id
        ).astype(float)
        N = np.sum(init_cond)  # total population
        model_specs = SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=N
        )
        # Forecast all of the components based on the forecasted beta
        forecasted_components = mr.forecast(
            model_specs=model_specs,
            init_cond=init_cond,
            times=times,
            betas=betas,
            dt=ode_fit_spec.parameters.solver_dt
        )
        forecasted_components[static_vars.COVARIATE_COL_DICT['COL_DATE']] = days

        data_interface.save_components(
            df=forecasted_components,
            location_id=location_id,
            draw_id=draw_id
        )

    data_interface.save_beta_scales(scales, location_id)


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

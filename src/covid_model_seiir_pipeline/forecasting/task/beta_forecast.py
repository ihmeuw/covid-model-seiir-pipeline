from argparse import ArgumentParser, Namespace
from typing import Optional
import logging
from pathlib import Path
import shlex

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline import static_vars

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
    # Thetas are a parameter generated from assumption or OOS predictive
    # validity testing to curtail some of the bad behavior of the model.
    thetas = data_interface.load_thetas(scenario_spec.theta)
    # Grab the last day of data in the model by location id.  This will
    # correspond to the initial condition for the projection.
    transition_date = data_interface.load_transition_date(draw_id)
    # we just want a map between location id and day we transition to
    # prediction.

    # We'll use the beta and SEIR compartments from this data set to get
    # the ODE initial condition.
    beta_regression_df = data_interface.load_beta_regression(draw_id).set_index('location_id').sort_index()
    past_components = beta_regression_df[['date', 'beta'] + static_vars.SEIIR_COMPARTMENTS]

    # Select out the initial condition using the day of transition.
    transition_day = past_components['date'] == transition_date.loc[past_components.index]
    initial_condition = past_components.loc[transition_day, static_vars.SEIIR_COMPARTMENTS]

    # Covariates and coefficients, and scaling parameters are
    # used to compute beta hat in the future.
    covariates = data_interface.load_covariates(scenario_spec, location_ids)
    coefficients = data_interface.load_regression_coefficients(draw_id)
    # Grab the projection of the covariates into the future, keeping the
    # day of transition from past model to future model.
    covariates = covariates.set_index('location_id').sort_index()
    the_future = covariates['date'] >= transition_date.loc[covariates.index]
    covariate_pred = covariates.loc[the_future].reset_index()

    beta_scales = data_interface.load_beta_scales(scenario=scenario_name, draw_id=draw_id)

    # We'll use the same params in the ODE forecast as we did in the fit.
    beta_params = data_interface.load_beta_params(draw_id=draw_id)

    # We'll need this to compute deaths and to splice with the forecasts.
    infection_data = data_interface.load_infection_data(draw_id)

    # Modeling starts

    betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
    future_components = model.run_normal_ode_model_by_location(initial_condition, beta_params, betas, thetas,
                                                               location_ids, scenario_spec.solver)
    components = model.splice_components(past_components, future_components, transition_date)
    components['theta'] = thetas.reindex(components.index).fillna(0)
    infections, deaths, r_effective = model.compute_output_metrics(infection_data, components, beta_params)

    if scenario_spec.algorithm == 'mandate_reimposition':
        min_wait = pd.Timedelta(days=7 * 2)
        days_on = pd.Timedelta(days=7 * 6)

        population = (components[static_vars.SEIIR_COMPARTMENTS]
                      .sum(axis=1)
                      .rename('population')
                      .groupby('location_id')
                      .max())
        death_rate = deaths.merge(population, on='location_id')
        death_rate['death_rate'] = death_rate['deaths'] / death_rate['population']
        mobility = covariates['mobility']
        percent_mandates = data_interface.load_covariate_info('mobility', 'mandate_lift', location_ids)
        mandate_effect = data_interface.load_covariate_info('mobility', 'effect', location_ids)
        reimposition_threshold = scenario_spec.algorithm_params['death_threshold']

        # compute reimposition dates
        over_threshold = death_rate['death_rate'] > reimposition_threshold
        projected = death_rate['observed'] == 0
        reimposition_date = death_rate[over_threshold & projected].groupy('location_id')['date'].min()
        min_reimposition_date = (transition_date + min_wait).loc[reimposition_date.index]
        reimposition_date = (reimposition_date
                             .where(reimposition_date >= min_reimposition_date, min_reimposition_date)
                             .rename('reimposition_date'))

        # compute mobility lower bound
        min_observed_mobility = mobility.groupby('location_id')['mobility'].min().rename('min_mobility')
        max_mandate_mobility = mandate_effect.sum(axis=1).rename('min_mobility')
        mobility_lower_bound = min_observed_mobility.where(min_observed_mobility <= max_mandate_mobility,
                                                           max_mandate_mobility)

        mobility_reference = (mobility
                              .merge(reimposition_date, how='left', on='location_id')
                              .merge(mobility_lower_bound, how='left', on='location_id'))

        reimposes = mobility_reference['reimposition_date'].notnull()
        dates_on = ((mobility_reference['reimposition_date'] <= mobility_reference['date'])
                    & mobility_reference['date'] <= mobility_reference['reimposition_date'] + min_wait)
        mobility_reference['mobility_explosion'] = mobility_reference['min_mobility'].where(reimposes & dates_on, np.nan)

        rampup = pd.merge(reimposition_date, percent_mandates, on='location_id', how='left')
        rampup['rampup'] = rampup.groupby('location_id')['percent'].apply(lambda x: x / x.max())
        rampup['first_date'] = rampup.groupby('location_id')['date'].transform('min')
        rampup['diff_date'] = rampup['reimposition_date'] - rampup['first_date']
        rampup['date'] = rampup['date'] + rampup['diff_date'] + days_on

        mobility_reference = mobility_reference.merge(rampup, how='left', on=['location_id', 'date'])
        post_explosion = mobility_reference['mobility_explosion'].isnull() & mobility_reference['rampup'].notnull()
        mobility_reference['mobility_explosion'] = mobility_reference['mobility_explosion'].where(
            ~post_explosion,
            mobility_reference['min_mobility'] * mobility_reference['rampup']
        )
        mobility_reference['mobility'] = mobility_reference[['mobility', 'mobility_explosion']].min(axis=1)

        covariate_pred['mobility'] = mobility_reference['mobility']

        betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
        future_components = model.run_normal_ode_model_by_location(initial_condition, beta_params, betas, thetas,
                                                                   location_ids, scenario_spec.solver)
        components = model.splice_components(past_components, future_components, transition_date)
        components['theta'] = thetas.reindex(components.index).fillna(0)
        infections, deaths, r_effective = model.compute_output_metrics(infection_data, components, beta_params)

    components = components.reset_index()
    outputs = pd.concat([infections, deaths, r_effective], axis=1).dropna().reset_index()

    data_interface.save_components(components, scenario_name, draw_id)
    data_interface.save_raw_outputs(outputs, scenario_name, draw_id)


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

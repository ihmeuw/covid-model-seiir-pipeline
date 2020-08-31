from argparse import ArgumentParser, Namespace
from typing import Optional
from pathlib import Path
import shlex

from covid_shared.cli_tools.logging import configure_logging_to_terminal
import pandas as pd
from loguru import logger

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting import model
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface




def run_beta_forecast(draw_id: int, forecast_version: str, scenario_name: str):
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario_name}, draw {draw_id}.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    logger.info('Loading input data.')
    location_ids = data_interface.load_location_ids()
    # Thetas are a parameter generated from assumption or OOS predictive
    # validity testing to curtail some of the bad behavior of the model.
    thetas = data_interface.load_thetas(scenario_spec.theta)
    # Grab the last day of data in the model by location id.  This will
    # correspond to the initial condition for the projection.
    transition_date = data_interface.load_transition_date(draw_id)

    # We'll use the beta and SEIR compartments from this data set to get
    # the ODE initial condition.
    beta_regression_df = data_interface.load_beta_regression(draw_id).set_index('location_id').sort_index()
    past_components = beta_regression_df[['date', 'beta'] + static_vars.SEIIR_COMPARTMENTS]

    # Select out the initial condition using the day of transition.
    transition_day = past_components['date'] == transition_date.loc[past_components.index]
    initial_condition = past_components.loc[transition_day, static_vars.SEIIR_COMPARTMENTS]
    before_model = past_components['date'] < transition_date.loc[past_components.index]
    past_components = past_components[before_model]

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
    logger.info('Forecasting beta and components.')
    betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
    future_components = model.run_normal_ode_model_by_location(initial_condition, beta_params, betas, thetas,
                                                               location_ids, scenario_spec.solver)
    logger.info('Processing ODE results and computing deaths and infections.')
    components = model.splice_components(past_components, future_components)
    components['theta'] = thetas.reindex(components.index).fillna(0)
    infections, deaths, r_effective = model.compute_output_metrics(infection_data, components, beta_params)

    if scenario_spec.algorithm == 'mandate_reimposition':
        logger.info('Entering mandate reimposition.')
        min_wait = pd.Timedelta(days=scenario_spec.algorithm_params['minimum_delay'])
        days_on = pd.Timedelta(days=static_vars.DAYS_PER_WEEK * scenario_spec.algorithm_params['reimposition_duration'])
        reimposition_threshold = scenario_spec.algorithm_params['death_threshold'] / 1e6

        population = (components[static_vars.SEIIR_COMPARTMENTS]
                      .sum(axis=1)
                      .rename('population')
                      .groupby('location_id')
                      .max())
        logger.info('Loading mandate reimposition data.')
        percent_mandates = data_interface.load_covariate_info('mobility', 'mandate_lift', location_ids)
        mandate_effect = data_interface.load_covariate_info('mobility', 'effect', location_ids)

        reimposition_count = 0
        reimposition_dates = {}
        reimposition_date = model.compute_reimposition_date(deaths, population, reimposition_threshold, min_wait)

        while len(reimposition_date):  # any place reimposes mandates.
            logger.info(f'On mandate reimposition {reimposition_count + 1}. {len(reimposition_date)} locations '
                        f'are reimposing mandates.')
            mobility = covariates[['date', 'mobility']].reset_index().set_index(['location_id', 'date'])['mobility']
            mobility_lower_bound = model.compute_mobility_lower_bound(mobility, mandate_effect)

            new_mobility = model.compute_new_mobility(mobility, reimposition_date,
                                                      mobility_lower_bound, percent_mandates,
                                                      min_wait, days_on)

            covariates = covariates.reset_index().set_index(['location_id', 'date'])
            covariates['mobility'] = new_mobility
            covariates = covariates.reset_index(level='date')
            covariate_pred = covariates.loc[the_future].reset_index()

            logger.info('Forecasting beta and components.')
            betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
            future_components = model.run_normal_ode_model_by_location(initial_condition, beta_params, betas, thetas,
                                                                       location_ids, scenario_spec.solver)
            logger.info('Processing ODE results and computing deaths and infections.')
            components = model.splice_components(past_components, future_components)
            components['theta'] = thetas.reindex(components.index).fillna(0)
            infections, deaths, r_effective = model.compute_output_metrics(infection_data, components, beta_params)

            reimposition_count += 1
            reimposition_dates[reimposition_count] = reimposition_date
            last_reimposition_end_date = reimposition_date + days_on
            reimposition_date = model.compute_reimposition_date(deaths, population, reimposition_threshold,
                                                                min_wait, last_reimposition_end_date)

    logger.info('Writing outputs.')
    components = components.reset_index()
    covariates = covariates.reset_index()
    outputs = pd.concat([infections, deaths, r_effective], axis=1).reset_index()

    data_interface.save_components(components, scenario_name, draw_id)
    data_interface.save_raw_covariates(covariates, scenario_name, draw_id)
    data_interface.save_raw_outputs(outputs, scenario_name, draw_id)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
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
    configure_logging_to_terminal(1)
    args = parse_arguments()
    run_beta_forecast(draw_id=args.draw_id,
                      forecast_version=args.forecast_version,
                      scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()

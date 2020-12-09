from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
import time
from typing import Optional


from covid_shared.cli_tools.logging import configure_logging_to_terminal
import pandas as pd
from loguru import logger

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting import model
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface


def run_beta_forecast(draw_id: int, forecast_version: str, scenario_name: str, **kwargs):
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario_name}, draw {draw_id}.")
    start = time.time()
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    _ode_time_s = 0  # Profiling variable

    logger.info('Loading input data.')
    location_ids = data_interface.load_location_ids()
    # We'll use the same params in the ODE forecast as we did in the fit.
    beta_params = data_interface.load_beta_params(draw_id=draw_id)
    # Thetas are a parameter generated from assumption or OOS predictive
    # validity testing to curtail some of the bad behavior of the model.
    thetas = data_interface.load_thetas(scenario_spec.theta)
    if ((1 < thetas) | thetas < -1).any():
        raise ValueError('Theta must be between -1 and 1.')
    if (beta_params['sigma'] - thetas >= 1).any():
        raise ValueError('Sigma - theta must be smaller than 1')

    # Grab the last day of data in the model by location id.  This will
    # correspond to the initial condition for the projection.
    transition_date = data_interface.load_transition_date(draw_id)

    # The population will be used to partition the SEIR compartments into
    # different sub groups for the forecast.
    population = data_interface.load_population()
    population_partition = model.get_population_partition(population,
                                                          location_ids,
                                                          scenario_spec.population_partition)

    # We'll use the beta and SEIR compartments from this data set to get
    # the ODE initial condition.
    beta_regression_df = data_interface.load_beta_regression(draw_id)
    compartment_info, past_components = model.get_past_components(
        beta_regression_df,
        population_partition,
        scenario_spec.system
    )
    # Select out the initial condition using the day of transition.
    transition_day = past_components['date'] == transition_date.loc[past_components.index]
    initial_condition = past_components.loc[transition_day, compartment_info.compartments]
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

    # We'll need this to compute deaths and to splice with the forecasts.
    infection_data = data_interface.load_infection_data(draw_id)
    ifr = data_interface.load_ifr_data()

    # Load any data specific to the particular scenario we're running
    scenario_data = data_interface.load_scenario_specific_data(location_ids, scenario_spec)

    # Modeling starts
    logger.info('Forecasting beta and components.')
    betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
    seir_parameters = model.prep_seir_parameters(betas, thetas, scenario_data)
    _ode_start = time.time()
    future_components = model.run_normal_ode_model_by_location(
        initial_condition,
        beta_params,
        seir_parameters,
        scenario_spec,
        compartment_info,
    )
    _ode_time_s += time.time() - _ode_start
    logger.info('Processing ODE results and computing deaths and infections.')
    components, infections, deaths, r_effective = model.compute_output_metrics(
        infection_data,
        ifr,
        past_components,
        future_components,
        beta_params,
        compartment_info,
    )

    if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
        logger.info('Entering mandate reimposition.')
        # Info data specific to mandate reimposition
        min_wait, days_on, reimposition_threshold = model.unpack_parameters(scenario_spec.algorithm_params)

        population = (components[compartment_info.compartments]
                      .sum(axis=1)
                      .rename('population')
                      .groupby('location_id')
                      .max())

        reimposition_count = 0
        reimposition_dates = {}
        last_reimposition_end_date = pd.Series(pd.NaT, index=population.index)
        reimposition_date = model.compute_reimposition_date(
            deaths,
            population,
            reimposition_threshold,
            min_wait,
            last_reimposition_end_date
        )

        while len(reimposition_date):  # any place reimposes mandates.
            logger.info(f'On mandate reimposition {reimposition_count + 1}. {len(reimposition_date)} locations '
                        f'are reimposing mandates.')
            mobility = covariates[['date', 'mobility']].reset_index().set_index(['location_id', 'date'])['mobility']
            mobility_lower_bound = model.compute_mobility_lower_bound(
                mobility,
                scenario_data.mandate_effects
            )

            new_mobility = model.compute_new_mobility(
                mobility,
                reimposition_date,
                mobility_lower_bound,
                scenario_data.percent_mandates,
                days_on
            )

            covariates = covariates.reset_index().set_index(['location_id', 'date'])
            covariates['mobility'] = new_mobility
            covariates = covariates.reset_index(level='date')
            covariate_pred = covariates.loc[the_future].reset_index()

            logger.info('Forecasting beta and components.')
            betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
            seir_parameters = model.prep_seir_parameters(betas, thetas, scenario_data)

            # The ode is done as a loop over the locations in the initial condition.
            # As locations that don't reimpose mandates produce identical forecasts,
            # subset here to only the locations that reimpose mandates for speed.
            initial_condition_subset = initial_condition.loc[reimposition_date.index]
            _ode_start = time.time()
            future_components_subset = model.run_normal_ode_model_by_location(
                initial_condition_subset,
                beta_params,
                seir_parameters,
                scenario_spec,
                compartment_info,
            )
            _ode_time_s += time.time() - _ode_start
            future_components = (future_components
                                 .drop(future_components_subset.index)
                                 .append(future_components_subset)
                                 .sort_index())

            logger.info('Processing ODE results and computing deaths and infections.')
            components, infections, deaths, r_effective = model.compute_output_metrics(
                infection_data,
                ifr,
                past_components,
                future_components,
                beta_params,
                compartment_info,
            )

            reimposition_count += 1
            reimposition_dates[reimposition_count] = reimposition_date
            last_reimposition_end_date.loc[reimposition_date.index] = reimposition_date + days_on
            reimposition_date = model.compute_reimposition_date(deaths, population, reimposition_threshold,
                                                                min_wait, last_reimposition_end_date)

    logger.info('Writing outputs.')
    components = components.reset_index()
    covariates = covariates.reset_index()
    outputs = pd.concat([infections, deaths, r_effective], axis=1).reset_index()

    data_interface.save_components(components, scenario_name, draw_id)
    data_interface.save_raw_covariates(covariates, scenario_name, draw_id)
    data_interface.save_raw_outputs(outputs, scenario_name, draw_id)
    logger.info(f'Total time: {time.time() - start}')
    logger.info(f'Total ODE time: {_ode_time_s}')


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)
    parser.add_argument("--extra-id", type=int, required=False)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)  # Debug level
    args = parse_arguments()
    run_beta_forecast(draw_id=args.draw_id,
                      forecast_version=args.forecast_version,
                      scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()

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


def run_beta_forecast(draw_id: int, forecast_version: str, scenario_name: str, **kwargs):
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

    if scenario_spec.algorithm == 'mean_level_mandate_reimposition':
        # Info data specific to mandate reimposition
        percent_mandates = data_interface.load_covariate_info('mobility', 'mandate_lift', location_ids)
        mandate_effect = data_interface.load_covariate_info('mobility', 'effect', location_ids)
        min_wait, days_on, reimposition_threshold = model.unpack_parameters(scenario_spec.algorithm_params)

        max_reimpositions = scenario_spec.algorithm_params['reimposition_count']
        reimposition_dates = []
        for reimposition_number in range(1, max_reimpositions + 1):
            try:
                df = data_interface.load_reimposition_dates(scenario_name, reimposition_number)
                df = df.set_index('location_id')['reimposition_date']
                df = pd.to_datetime(df)
                reimposition_dates.append(df)
            except FileNotFoundError:
                continue

        mobility = covariates[['date', 'mobility']].reset_index().set_index(['location_id', 'date'])['mobility']
        mobility_lower_bound = model.compute_mobility_lower_bound(mobility, mandate_effect)

        for reimposition_date in reimposition_dates:
            mobility = model.compute_new_mobility(mobility, reimposition_date,
                                                  mobility_lower_bound, percent_mandates, days_on).rename('mobility')

        covariates = covariates.reset_index().set_index(['location_id', 'date'])
        covariates['mobility'] = mobility
        covariates = covariates.reset_index(level='date')
        covariate_pred = covariates.loc[the_future].reset_index()

    beta_scales = data_interface.load_beta_scales(scenario=scenario_name, draw_id=draw_id)

    # We'll use the same params in the ODE forecast as we did in the fit.
    beta_params = data_interface.load_beta_params(draw_id=draw_id)

    # We'll need this to compute deaths and to splice with the forecasts.
    infection_data = data_interface.load_infection_data(draw_id)

    if scenario_spec.system == 'new_theta':
        if ((1 < thetas) | thetas < -1).any():
            raise ValueError('Theta must be between -1 and 1.')
        if (beta_params['sigma'] - thetas >= 1).any():
            raise ValueError('Sigma - theta must be smaller than 1')

    # Modeling starts
    logger.info('Forecasting beta and components.')
    betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
    future_components = model.run_normal_ode_model_by_location(initial_condition, beta_params, betas, thetas,
                                                               location_ids, scenario_spec.solver, scenario_spec.system)
    logger.info('Processing ODE results and computing deaths and infections.')
    components, infections, deaths, r_effective = model.compute_output_metrics(infection_data,
                                                                               past_components,
                                                                               future_components,
                                                                               thetas,
                                                                               beta_params,
                                                                               scenario_spec.system)

    if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
        logger.info('Entering mandate reimposition.')
        # Info data specific to mandate reimposition
        percent_mandates = data_interface.load_covariate_info('mobility', 'mandate_lift', location_ids)
        mandate_effect = data_interface.load_covariate_info('mobility', 'effect', location_ids)
        min_wait, days_on, reimposition_threshold = model.unpack_parameters(scenario_spec.algorithm_params)

        population = (components[static_vars.SEIIR_COMPARTMENTS]
                      .sum(axis=1)
                      .rename('population')
                      .groupby('location_id')
                      .max())
        logger.info('Loading mandate reimposition data.')

        reimposition_count = 0
        reimposition_dates = {}
        last_reimposition_end_date = pd.Series(pd.NaT, index=population.index)
        reimposition_date = model.compute_reimposition_date(deaths, population, reimposition_threshold,
                                                            min_wait, last_reimposition_end_date)

        while len(reimposition_date):  # any place reimposes mandates.
            logger.info(f'On mandate reimposition {reimposition_count + 1}. {len(reimposition_date)} locations '
                        f'are reimposing mandates.')
            mobility = covariates[['date', 'mobility']].reset_index().set_index(['location_id', 'date'])['mobility']
            mobility_lower_bound = model.compute_mobility_lower_bound(mobility, mandate_effect)

            new_mobility = model.compute_new_mobility(mobility, reimposition_date,
                                                      mobility_lower_bound, percent_mandates, days_on)

            covariates = covariates.reset_index().set_index(['location_id', 'date'])
            covariates['mobility'] = new_mobility
            covariates = covariates.reset_index(level='date')
            covariate_pred = covariates.loc[the_future].reset_index()

            logger.info('Forecasting beta and components.')
            betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
            future_components = model.run_normal_ode_model_by_location(initial_condition, beta_params,
                                                                       betas, thetas, location_ids,
                                                                       scenario_spec.solver, scenario_spec.system)
            logger.info('Processing ODE results and computing deaths and infections.')
            components, infections, deaths, r_effective = model.compute_output_metrics(infection_data,
                                                                                       past_components,
                                                                                       future_components,
                                                                                       thetas,
                                                                                       beta_params,
                                                                                       scenario_spec.system)

            reimposition_count += 1
            reimposition_dates[reimposition_count] = reimposition_date
            last_reimposition_end_date.loc[reimposition_date.index] = reimposition_date + days_on
            reimposition_date = model.compute_reimposition_date(deaths, population, reimposition_threshold,
                                                                min_wait, last_reimposition_end_date)

    logger.info('Writing outputs.')
    components = components.reset_index()
    covariates = covariates.reset_index()
    outputs = pd.concat([infections, deaths, r_effective], axis=1).reset_index()

    # If strict, don't allow overwriting of output files.
    # Mean level mandate reimposition will run and write these outputs
    # several times.
    strict = scenario_spec.algorithm != 'mean_level_mandate_reimposition'

    data_interface.save_components(components, scenario_name, draw_id, strict)
    data_interface.save_raw_covariates(covariates, scenario_name, draw_id, strict)
    data_interface.save_raw_outputs(outputs, scenario_name, draw_id, strict)


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

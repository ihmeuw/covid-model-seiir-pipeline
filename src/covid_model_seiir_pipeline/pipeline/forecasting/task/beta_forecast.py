from pathlib import Path

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.forecasting import model
from covid_model_seiir_pipeline.pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface


logger = cli_tools.task_performance_logger


def run_beta_forecast(forecast_version: str, scenario: str, draw_id: int, **kwargs):
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario}, draw {draw_id}.", context='setup')
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    logger.info('Loading input data.', context='read')
    location_ids = data_interface.load_location_ids()
    # We'll use the same params in the ODE forecast as we did in the fit.
    beta_params = data_interface.load_beta_params(draw_id=draw_id)
    # Thetas are a parameter generated from assumption or OOS predictive
    # validity testing to curtail some of the bad behavior of the model.
    thetas = data_interface.load_thetas(scenario_spec.theta, beta_params['sigma'])
    # Grab the last day of data in the model by location id.  This will
    # correspond to the initial condition for the projection.
    transition_date = data_interface.load_transition_date(draw_id)
    # The population will be used to partition the SEIR compartments into
    # different sub groups for the forecast.
    population = data_interface.load_five_year_population(location_ids)
    # We'll use the beta and SEIR compartments from this data set to get
    # the ODE initial condition.
    beta_regression_df = data_interface.load_beta_regression(draw_id)
    # Covariates and coefficients, and scaling parameters are
    # used to compute beta hat in the future.
    covariates = data_interface.load_covariates(scenario_spec, location_ids)
    coefficients = data_interface.load_regression_coefficients(draw_id)
    forecast_end_date = covariates.date.max()
    # Rescaling parameters for the beta forecast.
    beta_scales = data_interface.load_beta_scales(scenario=scenario, draw_id=draw_id)
    # Beta scale-up due to variant
    variant_prevalence = data_interface.load_variant_prevalence(
        scenario_spec.variant, transition_date, forecast_end_date
    )
    variant_max_beta_shift = scenario_spec.variant.get('beta_increase', 1)
    # noinspection PyTypeChecker
    variant_beta_shift = ((1 - variant_prevalence) + variant_max_beta_shift * variant_prevalence).rename('beta_pred')
    variant_max_vaccine_shift = scenario_spec.variant.get('vaccine_efficacy_decrease', 0)
    # noinspection PyTypeChecker
    variant_vaccine_shift = variant_max_vaccine_shift * variant_prevalence
    # We'll need this to compute deaths and to splice with the forecasts.
    infection_data = data_interface.load_infection_data(draw_id)
    ifr = data_interface.load_ifr_data(draw_id, location_ids)
    # Data for computing hospital usage
    mr = data_interface.load_mortality_ratio(location_ids)
    death_weights = model.get_death_weights(mr, population, with_error=False)
    hfr = data_interface.load_hospital_fatality_ratio(death_weights, location_ids)
    hospital_parameters = data_interface.get_hospital_parameters()
    correction_factors = data_interface.load_hospital_correction_factors()
    # Load any data specific to the particular scenario we're running
    scenario_data = data_interface.load_scenario_specific_data(location_ids, scenario_spec)

    logger.info('Processing inputs into model parameters.', context='transform')
    # Split the population into risk groups according to the specification.
    population_partition = model.get_population_partition(population, scenario_spec.population_partition)
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

    # Grab the projection of the covariates into the future, keeping the
    # day of transition from past model to future model.
    covariates = covariates.set_index('location_id').sort_index()
    the_future = covariates['date'] >= transition_date.loc[covariates.index]
    covariate_pred = covariates.loc[the_future].reset_index()
    betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
    betas = (betas.set_index('date', append=True).beta_pred * variant_beta_shift).reset_index(level='date')
    seir_parameters = model.prep_seir_parameters(betas, thetas, scenario_data, variant_vaccine_shift, population_partition)

    correction_factors = model.forecast_correction_factors(
        correction_factors,
        transition_date,
        forecast_end_date,
        hospital_parameters,
    )

    logger.info('Running ODE forecast.', context='compute_ode')
    future_components = model.run_normal_ode_model_by_location(
        initial_condition,
        beta_params,
        seir_parameters,
        scenario_spec,
        compartment_info,
    )
    logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
    output_metrics = model.compute_output_metrics(
        infection_data,
        ifr,
        past_components,
        future_components,
        beta_params,
        compartment_info,
    )
    hospital_usage = model.compute_corrected_hospital_usage(
        output_metrics.deaths,
        death_weights,
        hfr,
        hospital_parameters,
        correction_factors,
    )

    if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
        logger.info('Entering mandate reimposition.', context='compute_mandates')
        # Info data specific to mandate reimposition
        min_wait, days_on, reimposition_threshold, max_threshold = model.unpack_parameters(
            scenario_spec.algorithm_params,
            location_ids
        )
        reimposition_threshold = model.compute_reimposition_threshold(
            output_metrics.deaths,
            population,
            reimposition_threshold,
            max_threshold,
        )

        population = (output_metrics.components[compartment_info.compartments]
                      .sum(axis=1)
                      .rename('population')
                      .groupby('location_id')
                      .max())

        reimposition_count = 0
        reimposition_dates = {}
        last_reimposition_end_date = pd.Series(pd.NaT, index=population.index)
        reimposition_date = model.compute_reimposition_date(
            output_metrics.deaths,
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

            betas = model.forecast_beta(covariate_pred, coefficients, beta_scales)
            betas = (betas.set_index('date', append=True).beta_pred * variant_beta_shift).reset_index(level='date')
            seir_parameters = model.prep_seir_parameters(betas, thetas, scenario_data, variant_vaccine_shift,
                                                         population_partition)

            # The ode is done as a loop over the locations in the initial condition.
            # As locations that don't reimpose mandates produce identical forecasts,
            # subset here to only the locations that reimpose mandates for speed.
            initial_condition_subset = initial_condition.loc[reimposition_date.index]
            logger.info('Running ODE forecast.', context='compute_ode')
            future_components_subset = model.run_normal_ode_model_by_location(
                initial_condition_subset,
                beta_params,
                seir_parameters,
                scenario_spec,
                compartment_info,
            )

            logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
            future_components = (future_components
                                 .drop(future_components_subset.index)
                                 .append(future_components_subset)
                                 .sort_index())
            output_metrics = model.compute_output_metrics(
                infection_data,
                ifr,
                past_components,
                future_components,
                beta_params,
                compartment_info,
            )
            hospital_usage = model.compute_corrected_hospital_usage(
                output_metrics.deaths,
                death_weights,
                hfr,
                hospital_parameters,
                correction_factors,
            )

            logger.info('Recomputing reimposition dates', context='compute_mandates')
            reimposition_count += 1
            reimposition_dates[reimposition_count] = reimposition_date
            last_reimposition_end_date.loc[reimposition_date.index] = reimposition_date + days_on
            reimposition_date = model.compute_reimposition_date(
                output_metrics.deaths,
                population,
                reimposition_threshold,
                min_wait,
                last_reimposition_end_date,
            )

    logger.info('Writing outputs.', context='write')
    ode_param_cols = output_metrics.components.columns.difference(compartment_info.compartments)
    ode_params = output_metrics.components[ode_param_cols].reset_index()
    components = output_metrics.components[compartment_info.compartments].reset_index()
    covariates = covariates.reset_index()
    epi_metrics = [value for key, value in utilities.asdict(output_metrics).items() if key != 'components']
    usage = [value.rename(key) for key, value in utilities.asdict(hospital_usage).items()]
    corrections = [value.rename(key + '_correction') for key, value in utilities.asdict(correction_factors).items()]
    outputs = pd.concat(epi_metrics + usage + corrections, axis=1).reset_index()

    data_interface.save_ode_params(ode_params, scenario, draw_id)
    data_interface.save_components(components, scenario, draw_id)
    data_interface.save_raw_covariates(covariates, scenario, draw_id)
    data_interface.save_raw_outputs(outputs, scenario, draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_forecast_version
@cli_tools.with_scenario
@cli_tools.with_draw_id
@cli_tools.with_extra_id
@cli_tools.add_verbose_and_with_debugger
def beta_forecast(forecast_version: str, scenario: str, draw_id: int, extra_id: int,
                  verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)

    run = cli_tools.handle_exceptions(run_beta_forecast, logger, with_debugger)
    run(forecast_version=forecast_version,
        scenario=scenario,
        draw_id=draw_id)


if __name__ == '__main__':
    beta_forecast()

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


def run_beta_forecast(forecast_version: str, scenario: str, draw_id: int, progress_bar: bool):
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario}, draw {draw_id}.", context='setup')
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    #################
    # Build indices #
    #################
    # The hardest thing to keep consistent is data alignment. We have about 100
    # unique datasets in this model and they need to be aligned consistently
    # to do computation.
    logger.info('Loading index building data', context='read')
    past_infections = data_interface.load_past_infections(draw_id)
    past_start_dates = past_infections.reset_index().groupby('location_id').date.min()
    forecast_start_dates = past_infections.reset_index().groupby('location_id').date.max()
    # Forecast is run to the end of the covariates
    covariates = data_interface.load_covariates(scenario_spec)
    forecast_end_dates = covariates.reset_index().groupby('location_id').date.max()
    logger.info('Building indices', context='transform')
    indices = model.Indices(
        past_start_dates,
        forecast_start_dates,
        forecast_end_dates,
    )

    ########################################
    # Build parameters for the SEIIR model #
    ########################################
    logger.info('Loading SEIIR parameter input data.', context='read')
    # We'll use the same params in the ODE forecast as we did in the fit.
    ode_params = data_interface.load_ode_parameters(draw_id=draw_id)
    # Contains both the fit and regression betas
    betas = data_interface.load_betas(draw_id)
    # Thetas are a parameter generated from assumption or OOS predictive
    # validity testing to curtail some of the bad behavior of the model.
    thetas = data_interface.load_thetas(scenario_spec.theta)
    # Regression coefficients for forecasting beta.
    coefficients = data_interface.load_coefficients(draw_id)
    # Rescaling parameters for the beta forecast.
    beta_scales = data_interface.load_beta_scales(scenario=scenario, draw_id=draw_id)
    # Vaccine data, of course.
    vaccinations = data_interface.load_vaccinations(scenario_spec.vaccine_version)
    # Collate all the parameters, ensure consistent index, etc.
    logger.info('Processing inputs into model parameters.', context='transform')
    covariates = covariates.reindex(indices.full)
    model_parameters = model.build_model_parameters(
        indices,
        ode_params,
        betas,
        thetas,
        covariates,
        coefficients,
        beta_scales,
        vaccinations,
        scenario_spec,
    )

    ##################################
    # Redistribute  past compartments#
    ##################################
    logger.info('Loading past compartment data.', context='read')
    compartments = data_interface.load_compartments(draw_id=draw_id)
    population = data_interface.load_five_year_population()
    logger.info('Redistributing past compartments.', context='transform')
    compartments = model.redistribute_past_compartments(
        indices=indices,
        compartments=compartments,
        population=population,
        model_parameters=model_parameters,
    )







    logger.info('Running ODE forecast.', context='compute_ode')
    future_components = model.run_normal_ode_model_by_location(
        initial_condition,
        beta_params,
        seiir_parameters,
        scenario_spec,
        compartment_info,
    )
    logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
    output_metrics = model.compute_output_metrics(
        infection_data,
        ratio_data,
        past_components,
        future_components,
        beta_params,
        compartment_info,
    )
    hospital_usage = model.compute_corrected_hospital_usage(
        output_metrics.admissions,
        ratio_data.ihr / ratio_data.ifr,
        hospital_parameters,
        correction_factors,
    )

    if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
        logger.info('Entering mandate reimposition.', context='compute_mandates')
        # Info data specific to mandate reimposition
        location_ids = data_interface.load_location_ids()
        min_wait, days_on, reimposition_threshold, max_threshold = model.unpack_parameters(
            scenario_spec.algorithm_params,
            location_ids
        )
        population = (output_metrics.components[compartment_info.compartments]
                      .sum(axis=1)
                      .rename('population')
                      .groupby('location_id')
                      .max())
        reimposition_threshold = model.compute_reimposition_threshold(
            output_metrics.deaths,
            population,
            reimposition_threshold,
            max_threshold,
        )
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
            seiir_parameters = model.prep_seiir_parameters(
                betas,
                thetas,
                scenario_data,
            )

            # The ode is done as a loop over the locations in the initial condition.
            # As locations that don't reimpose mandates produce identical forecasts,
            # subset here to only the locations that reimpose mandates for speed.
            initial_condition_subset = initial_condition.loc[reimposition_date.index]
            logger.info('Running ODE forecast.', context='compute_ode')
            future_components_subset = model.run_normal_ode_model_by_location(
                initial_condition_subset,
                beta_params,
                seiir_parameters,
                scenario_spec,
                compartment_info,
            ).set_index('date', append=True).sort_index()

            logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
            future_components = (future_components
                                 .set_index('date', append=True)
                                 .sort_index()
                                 .drop(future_components_subset.index)
                                 .append(future_components_subset)
                                 .sort_index()
                                 .reset_index(level='date'))
            output_metrics = model.compute_output_metrics(
                infection_data,
                ratio_data,
                past_components,
                future_components,
                beta_params,
                compartment_info,
            )
            hospital_usage = model.compute_corrected_hospital_usage(
                output_metrics.admissions,
                ratio_data.ihr / ratio_data.ifr,
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
    ode_params = output_metrics.components[ode_param_cols]
    components = output_metrics.components[compartment_info.compartments]
    covariates = covariates.set_index('date', append=True)
    epi_metrics = [value for key, value in utilities.asdict(output_metrics).items() if key != 'components']
    usage = [value.rename(key) for key, value in utilities.asdict(hospital_usage).items()]
    corrections = [value.rename(key + '_correction') for key, value in utilities.asdict(correction_factors).items()]
    outputs = pd.concat(epi_metrics + usage + corrections, axis=1)

    data_interface.save_ode_params(ode_params, scenario, draw_id)
    data_interface.save_components(components, scenario, draw_id)
    data_interface.save_raw_covariates(covariates, scenario, draw_id)
    data_interface.save_raw_outputs(outputs, scenario, draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_forecast_version
@cli_tools.with_scenario
@cli_tools.with_draw_id
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def beta_forecast(forecast_version: str, scenario: str, draw_id: int,
                  progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)

    run = cli_tools.handle_exceptions(run_beta_forecast, logger, with_debugger)
    run(forecast_version=forecast_version,
        scenario=scenario,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    beta_forecast()

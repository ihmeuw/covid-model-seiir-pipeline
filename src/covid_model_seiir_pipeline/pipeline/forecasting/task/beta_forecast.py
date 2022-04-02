import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.forecasting import model
from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
    ForecastSpecification,
    FORECAST_JOBS,
)
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface


logger = cli_tools.task_performance_logger


def run_beta_forecast(forecast_version: str, scenario: str, draw_id: int, progress_bar: bool):
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario}, draw {draw_id}.", context='setup')
    specification = ForecastSpecification.from_version_root(forecast_version)
    num_cores = specification.workflow.task_specifications[FORECAST_JOBS.forecast].num_cores
    scenario_spec = specification.scenarios[scenario]
    data_interface = ForecastDataInterface.from_specification(specification)

    #################
    # Build indices #
    #################
    # The hardest thing to keep consistent is data alignment. We have about 100
    # unique datasets in this model, and they need to be aligned consistently
    # to do computation.
    logger.info('Loading index building data', context='read')
    location_ids = data_interface.load_location_ids()
    past_compartments = data_interface.load_past_compartments(draw_id).loc[location_ids]
    past_compartments = past_compartments.loc[past_compartments.notnull().any(axis=1)]
    # Contains both the fit and regression betas
    betas = data_interface.load_regression_beta(draw_id).loc[location_ids]
    ode_params = data_interface.load_fit_ode_params(draw_id=draw_id)
    durations = ode_params.filter(like='exposure').iloc[0]
    epi_data = data_interface.load_input_epi_measures(draw_id=draw_id).loc[location_ids]

    past_start_dates = past_compartments.reset_index(level='date').date.groupby('location_id').min()
    beta_fit_end_dates = betas['beta'].dropna().reset_index(level='date').date.groupby('location_id').max()

    # We want the forecast to start at the last date for which all reported measures
    # with at least one report in the location are present.
    past_compartments = past_compartments.reset_index()
    measure_dates = []
    for measure in ['case', 'death', 'admission']:
        duration = durations.at[f'exposure_to_{measure}']
        epi_measure = {'case': 'cases', 'death': 'deaths', 'admission': 'hospitalizations'}[measure]
        dates = (epi_data[f'smoothed_daily_{epi_measure}']
                 .groupby('location_id')
                 .shift(-duration)
                 .dropna()
                 .reset_index()
                 .groupby('location_id')
                 .date
                 .max())
        measure_dates.append(dates)
        cols = [c for c in past_compartments.columns if measure.capitalize() in c]
        for location_id, date in dates.iteritems():
            past_compartments.loc[((past_compartments.location_id == location_id) 
                                  & (past_compartments.date > date)), cols] = np.nan

    forecast_start_dates = pd.concat([beta_fit_end_dates, *measure_dates], axis=1).min(axis=1).rename('date')
    past_compartments = past_compartments.set_index(['location_id', 'date'])

    # Forecast is run to the end of the covariates
    covariates = data_interface.load_covariates(scenario_spec.covariates)
    forecast_end_dates = covariates.reset_index().groupby('location_id').date.max()
    population = data_interface.load_population('total').population

    logger.info('Building indices', context='transform')
    indices = model.Indices(
        past_start_dates,
        beta_fit_end_dates,
        forecast_start_dates,
        forecast_end_dates,
    )

    ########################################
    # Build parameters for the SEIIR model #
    ########################################
    logger.info('Loading SEIIR parameter input data.', context='read')    
    # Use to get ratios
    posterior_epi_measures = data_interface.load_posterior_epi_measures(draw_id=draw_id)
    prior_ratios = data_interface.load_rates(draw_id).loc[location_ids]    
    # Rescaling parameters for the beta forecast.
    beta_shift_parameters = data_interface.load_beta_scales(scenario=scenario, draw_id=draw_id)
    # Regression coefficients for forecasting beta.
    coefficients = data_interface.load_coefficients(draw_id)
    # Vaccine data, of course.
    vaccinations = data_interface.load_vaccine_uptake(scenario_spec.vaccine_version)
    etas = data_interface.load_vaccine_risk_reduction(scenario_spec.vaccine_version)
    phis = data_interface.load_phis(draw_id=draw_id)
    # Variant prevalences.
    rhos = data_interface.load_variant_prevalence(scenario_spec.variant_version)

    hospital_cf = data_interface.load_hospitalizations(measure='correction_factors')
    hospital_parameters = data_interface.get_hospital_params()

    log_beta_shift = (scenario_spec.log_beta_shift,
                      pd.Timestamp(scenario_spec.log_beta_shift_date))
    beta_scale = (scenario_spec.beta_scale,
                  pd.Timestamp(scenario_spec.beta_scale_date))

    risk_group_population = data_interface.load_population('risk_group')
    risk_group_population = risk_group_population.divide(risk_group_population.sum(axis=1), axis=0)

    # Collate all the parameters, ensure consistent index, etc.
    logger.info('Processing inputs into model parameters.', context='transform')
    covariates = covariates.reindex(indices.full)
    beta, beta_hat = model.build_beta_final(
        indices,
        betas,
        covariates,
        coefficients,
        beta_shift_parameters,
        log_beta_shift,
        beta_scale,
    )
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    logger.warning('Using Hong Kong IFR projection for mainland China IFR projection in `ode_forecast.build_ratio`.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    model_parameters = model.build_model_parameters(
        indices,
        beta,
        past_compartments,
        prior_ratios,
        ode_params,
        rhos,
        vaccinations,
        etas,
        phis,
        risk_group_population,
    )
    hospital_cf = model.forecast_correction_factors(
        indices,
        correction_factors=hospital_cf,
        hospital_parameters=hospital_parameters,
    )

    # Pull in compartments from the fit and subset out the initial condition.
    logger.info('Loading past compartment data.', context='read')
    initial_condition = past_compartments.loc[indices.past].reindex(indices.full, fill_value=0.)
    initial_condition[initial_condition < 0.] = 0.

    logger.info('Running ODE forecast.', context='compute_ode')
    compartments, chis = model.run_ode_forecast(
        initial_condition,
        model_parameters,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    exposure_to_death = ode_params['exposure_to_death'].iloc[0]
    past_deaths = (compartments
                   .filter(like='Death_all_all_all')
                   .sum(axis=1)
                   .loc[indices.past]
                   .groupby('location_id')
                   .apply(lambda x: x.reset_index(level=0, drop=True)
                                     .shift(exposure_to_death, freq='D'))
                   .rename('value'))
    total_deaths = (compartments
                    .filter(like='Death_all_all_all')
                    .sum(axis=1)
                    .groupby('location_id')
                    .apply(lambda x: x.reset_index(level=0, drop=True)
                                      .shift(exposure_to_death, freq='D'))
                    .rename('value')
                    .to_frame())
    total_deaths['observed'] = 0
    total_deaths.loc[past_deaths.index, 'observed'] = 1
    total_deaths = total_deaths.set_index('observed', append=True).value

    if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
        logger.info('Entering mandate reimposition.', context='compute_mandates')
        # Info data specific to mandate reimposition
        percent_mandates, mandate_effects = data_interface.load_mandate_data(scenario_spec.covariates['mobility'])
        mortality_scalars = data_interface.load_total_covid_scalars(draw_id).loc[location_ids, 'scalar']
        mortality_scalars = (mortality_scalars
                             .reindex(indices.full)
                             .groupby('location_id')
                             .ffill()
                             .groupby('location_id')
                             .bfill())
        min_wait, days_on, reimposition_threshold, max_threshold = model.unpack_parameters(
            scenario_spec.algorithm_params,
            mortality_scalars,
        )
        reimposition_threshold = model.compute_reimposition_threshold(
            past_deaths,
            population,
            reimposition_threshold,
            max_threshold,
        )
        reimposition_count = 0
        reimposition_dates = {}
        last_reimposition_end_date = pd.Series(pd.NaT, index=population.index)
        reimposition_date = model.compute_reimposition_date(
            total_deaths,
            population,
            reimposition_threshold,
            min_wait,
            last_reimposition_end_date
        )

        while len(reimposition_date):  # any place reimposes mandates.
            logger.info(f'On mandate reimposition {reimposition_count + 1}. {len(reimposition_date)} locations '
                        f'are reimposing mandates.')
            mobility = covariates['mobility']
            mobility_lower_bound = model.compute_mobility_lower_bound(
                mobility,
                mandate_effects,
            )

            new_mobility = model.compute_new_mobility(
                mobility,
                reimposition_date,
                mobility_lower_bound,
                percent_mandates,
                days_on,
            )

            covariates['mobility'] = new_mobility

            beta, beta_hat = model.build_beta_final(
                indices,
                betas,
                covariates,
                coefficients,
                beta_shift_parameters,
                log_beta_shift,
                beta_scale,
            )
            model_parameters = model.build_model_parameters(
                indices,
                beta,
                past_compartments,
                prior_ratios,
                ode_params,
                rhos,
                vaccinations,
                etas,
                phis,
            )

            # The ode is done as a loop over the locations in the initial condition.
            # As locations that don't reimpose mandates produce identical forecasts,
            # subset here to only the locations that reimpose mandates for speed.
            initial_condition_subset = initial_condition.loc[reimposition_date.index]
            logger.info('Running ODE forecast.', context='compute_ode')
            compartments_subset, chis = model.run_ode_forecast(
                initial_condition_subset,
                model_parameters,
                num_cores=num_cores,
                progress_bar=progress_bar,
            )

            logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
            compartments = (compartments
                            .sort_index()
                            .drop(compartments_subset.index)
                            .append(compartments_subset)
                            .sort_index())
            total_deaths = (compartments
                            .filter(like='Death_all_all_all')
                            .sum(axis=1)
                            .groupby('location_id')
                            .apply(lambda x: x.reset_index(level=0, drop=True)
                                              .shift(exposure_to_death, freq='D'))
                            .rename('value')
                            .to_frame())
            total_deaths['observed'] = 0
            total_deaths.loc[past_deaths.index, 'observed'] = 1
            total_deaths = total_deaths.set_index('observed', append=True).value

            logger.info('Recomputing reimposition dates', context='compute_mandates')
            reimposition_count += 1
            reimposition_dates[reimposition_count] = reimposition_date
            last_reimposition_end_date.loc[reimposition_date.index] = reimposition_date + days_on
            reimposition_date = model.compute_reimposition_date(
                total_deaths,
                population,
                reimposition_threshold,
                min_wait,
                last_reimposition_end_date,
            )

    system_metrics = model.compute_output_metrics(
        indices,
        compartments,
        model_parameters,
        ode_params,
        hospital_parameters,
        hospital_cf,
    )

    logger.info('Prepping outputs.', context='transform')
    forecast_ode_params = pd.concat([model_parameters.base_parameters, beta, beta_hat], axis=1)
    for measure in ['death', 'case', 'admission']:
        forecast_ode_params[f'exposure_to_{measure}'] = ode_params[f'exposure_to_{measure}'].iloc[0]

    logger.info('Writing outputs.', context='write')
    data_interface.save_ode_params(forecast_ode_params, scenario, draw_id)
    data_interface.save_components(compartments, scenario, draw_id)
    data_interface.save_raw_covariates(covariates, scenario, draw_id)
    data_interface.save_raw_outputs(system_metrics, scenario, draw_id)

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

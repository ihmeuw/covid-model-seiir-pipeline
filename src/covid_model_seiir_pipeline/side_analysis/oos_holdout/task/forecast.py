import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.forecasting import model
from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import (
    OOSHoldoutSpecification,
    OOS_HOLDOUT_JOBS,
)
from covid_model_seiir_pipeline.side_analysis.oos_holdout.data import (
    OOSHoldoutDataInterface,
)


logger = cli_tools.task_performance_logger


def run_oos_forecast(oos_holdout_version: str, draw_id: int, progress_bar: bool):
    logger.info(f"Initiating OOS holdout forecast for draw {draw_id}.", context='setup')
    specification = OOSHoldoutSpecification.from_version_root(oos_holdout_version)
    num_cores = specification.workflow.task_specifications[OOS_HOLDOUT_JOBS.oos_forecast].num_cores
    data_interface = OOSHoldoutDataInterface.from_specification(specification)

    scenario = specification.data.seir_forecast_scenario
    forecast_specification = data_interface.forecast_data_interface.load_specification()
    scenario_spec = forecast_specification.scenarios[scenario]

    #################
    # Build indices #
    #################
    # The hardest thing to keep consistent is data alignment. We have about 100
    # unique datasets in this model, and they need to be aligned consistently
    # to do computation.
    logger.info('Loading index building data', context='read')
    location_ids = data_interface.load_location_ids()
    holdout_days = pd.Timedelta(days=specification.parameters.holdout_weeks * 7)
    past_compartments = data_interface.load_past_compartments(draw_id).loc[location_ids]
    past_compartments = past_compartments.loc[past_compartments.notnull().any(axis=1)]
    # Contains both the fit and regression betas
    betas = data_interface.load_regression_beta(draw_id).loc[location_ids]
    ode_params = data_interface.load_fit_ode_params(draw_id=draw_id)
    durations = ode_params.filter(like='exposure').iloc[0]
    epi_data = data_interface.load_input_epi_measures(draw_id=draw_id).loc[location_ids]

    past_start_dates = past_compartments.reset_index(level='date').date.groupby('location_id').min()
    beta_fit_end_dates = betas['beta'].dropna().reset_index(level='date').date.groupby('location_id').max() - holdout_days

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
    covariates = data_interface.load_covariates()
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

    full_outputs = data_interface.load_raw_outputs(scenario, draw_id)
    delta = system_metrics - full_outputs

    logger.info('Writing outputs.', context='write')
    data_interface.save_raw_oos_outputs(system_metrics, draw_id)
    data_interface.save_deltas(delta, draw_id)


    logger.report()


@click.command()
@cli_tools.with_task_oos_holdout_version
@cli_tools.with_draw_id
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def oos_forecast(oos_holdout_version: str,
                 draw_id: int,
                 progress_bar: bool,
                 verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)

    run = cli_tools.handle_exceptions(run_oos_forecast, logger, with_debugger)
    run(oos_holdout_version=oos_holdout_version,
        draw_id=draw_id,
        progress_bar=progress_bar)

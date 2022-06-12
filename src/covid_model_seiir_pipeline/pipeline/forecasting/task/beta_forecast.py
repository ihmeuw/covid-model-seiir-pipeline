import click
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
    hierarchy = data_interface.load_hierarchy('pred')
    past_compartments = data_interface.load_past_compartments(draw_id).loc[location_ids]
    past_compartments = past_compartments.loc[past_compartments.notnull().any(axis=1)]
    # Contains both the fit and regression betas
    betas = data_interface.load_regression_beta(draw_id).loc[location_ids]
    ode_params = data_interface.load_fit_ode_params(draw_id=draw_id)
    epi_data = data_interface.load_input_epi_measures(draw_id=draw_id).loc[location_ids]
    covariates = data_interface.load_covariates(scenario_spec.covariates)

    logger.info('Building indices', context='transform')
    indices = model.build_indices(
        betas=betas,
        ode_params=ode_params,
        past_compartments=past_compartments,
        epi_data=epi_data,
        covariates=covariates,
    )

    ########################################
    # Build parameters for the SEIIR model #
    ########################################
    logger.info('Loading SEIIR parameter input data.', context='read')
    population = data_interface.load_population('total').population
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
    antiviral_risk_reduction = model.build_antiviral_risk_reduction(
        index=indices.full,
        hierarchy=hierarchy,
        scenario_spec=scenario_spec.antiviral_specification,
    )
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    logger.warning('Using Hong Kong IFR projection for mainland China IFR projection in `ode_forecast.build_ratio`.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    hierarchy = data_interface.load_hierarchy('pred')
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
        antiviral_risk_reduction,
        risk_group_population,
        hierarchy,
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

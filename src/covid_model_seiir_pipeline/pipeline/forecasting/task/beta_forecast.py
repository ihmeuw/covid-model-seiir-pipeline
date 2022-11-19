import functools

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
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario}, draw {draw_id}.",
                context='setup')
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
    hierarchy = data_interface.load_hierarchy('pred')
    location_ids = data_interface.load_location_ids()
    past_compartments = data_interface.load_past_compartments(draw_id).loc[location_ids]
    ode_params = data_interface.load_fit_ode_params(draw_id=draw_id)
    epi_data = data_interface.load_input_epi_measures(draw_id=draw_id).loc[location_ids]
    mortality_scalars = data_interface.load_total_covid_scalars(draw_id).scalar.loc[
        location_ids]

    # We want the forecast to start at the last date for which all reported measures
    # with at least one report in the location are present.
    past_compartments, measure_dates = model.filter_past_compartments(
        past_compartments=past_compartments,
        ode_params=ode_params,
        epi_data=epi_data,
    )
    # Contains both the fit and regression betas
    betas = data_interface.load_regression_beta(draw_id).loc[location_ids]
    covariates = data_interface.load_covariates(scenario_spec.covariates)

    logger.info('Building indices', context='transform')
    indices = model.build_indices(
        betas=betas,
        past_compartments=past_compartments,
        measure_dates=measure_dates,
        covariates=covariates,
    )

    covariates = covariates.reindex(indices.full)

    ########################################
    # Build parameters for the SEIIR model #
    ########################################
    logger.info('Loading SEIIR parameter input data.', context='read')
    # Use to get ratios
    prior_ratios = data_interface.load_rates(draw_id).loc[location_ids]
    # Rescaling parameters for the beta forecast.
    beta_shift_parameters = data_interface.load_beta_scales(scenario=scenario,
                                                            draw_id=draw_id)
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

    antiviral_coverage = data_interface.load_antiviral_coverage(
        scenario=scenario_spec.antiviral_version)
    antiviral_effectiveness = data_interface.load_antiviral_effectiveness(draw_id=draw_id)
    antiviral_rr = model.compute_antiviral_rr(antiviral_coverage, antiviral_effectiveness)

    total_population = data_interface.load_population('total').population
    risk_group_population = data_interface.load_population('risk_group')
    risk_group_population = risk_group_population.divide(risk_group_population.sum(axis=1), axis=0)

    initial_condition = past_compartments.loc[indices.past].reindex(indices.full, fill_value=0.)
    initial_condition[initial_condition < 0.] = 0.

    # Collate all the parameters, ensure consistent index, etc.
    logger.info('Processing inputs into model parameters.', context='transform')

    min_reimposition_dates = (indices.future
                              .to_frame()
                              .reset_index(drop=True)
                              .groupby('location_id')
                              .date.min() + pd.Timedelta(days=14))
    reimposition_threshold = model.get_reimposition_threshold(
        indices=indices,
        population=total_population,
        hierarchy=hierarchy,
        epi_data=epi_data,
        rhos=rhos,
        mortality_scalars=mortality_scalars,
        reimposition_params=scenario_spec.mandate_reimposition,
    )
    reimposition_levels = model.get_reimposition_levels(
        covariates=covariates,
        rhos=rhos,
        hierarchy=hierarchy,
    )

    hospital_cf = model.forecast_correction_factors(
        indices,
        correction_factors=hospital_cf,
        hospital_parameters=hospital_parameters,
    )

    initial_condition = past_compartments.loc[indices.past].reindex(indices.full,
                                                                    fill_value=0.)
    initial_condition[initial_condition < 0.] = 0.

    build_beta_final = functools.partial(
        model.build_beta_final,
        indices=indices,
        beta_regression=betas,
        coefficients=coefficients,
        beta_shift_parameters=beta_shift_parameters,
    )

    beta, beta_hat = build_beta_final(covariates=covariates)
    model_parameters = model.build_model_parameters(
        indices=indices,
        beta=beta,
        past_compartments=past_compartments,
        prior_ratios=prior_ratios,
        rates_projection_spec=scenario_spec.rates_projection,
        ode_parameters=ode_params,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        phis=phis,
        antiviral_rr=antiviral_rr,
        risk_group_population=risk_group_population,
        hierarchy=hierarchy,
    )

    reimposition_number = 0
    max_num_reimpositions = scenario_spec.mandate_reimposition['max_num_reimpositions']
    locations_to_run = list(initial_condition.reset_index().location_id.unique())
    done_data = []
    while reimposition_number <= max_num_reimpositions and locations_to_run:
        logger.info(
            f'Running ODE system on reimposition {reimposition_number} for {len(locations_to_run)} locations.',
            context='ODE system'
        )
        compartments, chis, failed = model.run_ode_forecast(
            initial_condition,
            model_parameters,
            num_cores=num_cores,
            progress_bar=progress_bar,
            location_ids=locations_to_run,
        )

        reimposition_dates = model.compute_reimposition_dates(
            compartments=compartments,
            total_population=total_population,
            min_reimposition_dates=min_reimposition_dates,
            reimposition_threshold=reimposition_threshold,
        )

        locations_to_run = reimposition_dates.index.tolist()
        if not locations_to_run:
            break

        done_data.append(compartments.drop(locations_to_run, level='location_id'))

        covariates, min_reimposition_dates = model.reimpose_mandates(
            reimposition_dates=reimposition_dates,
            reimposition_levels=reimposition_levels,
            covariates=covariates,
            min_reimposition_dates=min_reimposition_dates,
        )

        beta, beta_hat = build_beta_final(covariates=covariates)
        model_parameters.base_parameters.loc[:, 'beta_all_infection'] = beta
        reimposition_number += 1

    compartments = pd.concat(done_data)

    system_metrics = model.compute_output_metrics(
        indices=indices,
        ode_params=ode_params,
        hospital_parameters=hospital_parameters,
        hospital_cf=hospital_cf,
        compartments=compartments,
        model_parameters=model_parameters,
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

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.forecasting import model
from covid_model_seiir_pipeline.pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface


logger = cli_tools.task_performance_logger


def run_beta_forecast(forecast_version: str, scenario: str, draw_id: int, progress_bar: bool):
    logger.info(f"Initiating SEIIR beta forecasting for scenario {scenario}, draw {draw_id}.", context='setup')
    forecast_spec: ForecastSpecification = ForecastSpecification.from_version_root(forecast_version)
    scenario_spec = forecast_spec.scenarios[scenario]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    #################
    # Build indices #
    #################
    # The hardest thing to keep consistent is data alignment. We have about 100
    # unique datasets in this model, and they need to be aligned consistently
    # to do computation.
    logger.info('Loading index building data', context='read')
    past_infections = data_interface.load_posterior_epi_measures(draw_id, columns=['daily_total_infections'])
    past_start_dates = past_infections.reset_index().groupby('location_id').date.min()
    forecast_start_dates = past_infections.reset_index().groupby('location_id').date.max()
    # Forecast is run to the end of the covariates
    covariates = data_interface.load_covariates(scenario_spec.covariates)
    forecast_end_dates = covariates.reset_index().groupby('location_id').date.max()
    population = data_interface.load_population('total').population

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
    ode_params = data_interface.load_regression_ode_params(draw_id=draw_id)
    # Contains both the fit and regression betas
    betas = data_interface.load_regression_beta(draw_id)
    # Rescaling parameters for the beta forecast.
    beta_shift_parameters = data_interface.load_beta_scales(scenario=scenario, draw_id=draw_id)
    # Regression coefficients for forecasting beta.
    coefficients = data_interface.load_coefficients(draw_id)
    # Vaccine data, of course.
    vaccinations = data_interface.load_vaccine_uptake(scenario_spec.vaccine_version)
    etas = data_interface.load_vaccine_risk_reduction(scenario_spec.vaccine_version)
    natural_waning_dist = data_interface.load_waning_parameters(scenario_spec.vaccine_version)
    natural_waning_matrix = data_interface.load_cross_variant_immunity_matrix(scenario_spec.vaccine_version)

    # Variant prevalences.
    rhos = data_interface.load_variant_prevalence(scenario_spec.variant_version)
    log_beta_shift = (scenario_spec.log_beta_shift,
                      pd.Timestamp(scenario_spec.log_beta_shift_date))
    beta_scale = (scenario_spec.beta_scale,
                  pd.Timestamp(scenario_spec.beta_scale_date))

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
    model_parameters = model.build_model_parameters(
        indices,
        beta,
        ode_params,
        rhos,
        vaccinations,
        etas,
        natural_waning_dist,
        natural_waning_matrix,
    )

    # Pull in compartments from the fit and subset out the initial condition.
    logger.info('Loading past compartment data.', context='read')
    past_compartments = data_interface.load_compartments(draw_id=draw_id)
    initial_condition = past_compartments.reindex(indices.full, fill_value=0.)

    ###################################################
    # Construct parameters for postprocessing results #
    ###################################################
    logger.info('Loading results processing input data.', context='read')
    past_deaths = data_interface.load_past_deaths(draw_id=draw_id).dropna()
    ratio_data = data_interface.load_ratio_data(draw_id=draw_id)
    hospital_parameters = data_interface.get_hospital_parameters()
    correction_factors = data_interface.load_hospital_correction_factors()

    logger.info('Prepping results processing parameters.', context='transform')
    postprocessing_params = model.build_postprocessing_parameters(
        indices,
        past_infections,
        past_deaths,
        ratio_data,
        model_parameters,
        correction_factors,
        hospital_parameters,
    )

    logger.info('Running ODE forecast.', context='compute_ode')
    compartments, chis = model.run_ode_forecast(
        initial_condition,
        model_parameters,
    )
    logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
    system_metrics, output_metrics = model.compute_output_metrics(
        indices,
        compartments,
        postprocessing_params,
        model_parameters,
        hospital_parameters,
    )

    if scenario_spec.algorithm == 'draw_level_mandate_reimposition':
        logger.info('Entering mandate reimposition.', context='compute_mandates')
        # Info data specific to mandate reimposition
        percent_mandates, mandate_effects = data_interface.load_mandate_data(scenario_spec.covariates['mobility'])
        em_scalars = data_interface.load_em_scalars(draw_id)
        min_wait, days_on, reimposition_threshold, max_threshold = model.unpack_parameters(
            scenario_spec.algorithm_params,
            em_scalars,
        )
        reimposition_threshold = model.compute_reimposition_threshold(
            postprocessing_params.past_deaths,
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
                ode_params,
                rhos,
                vaccinations,
                etas,
                natural_waning_dist,
                natural_waning_matrix,
            )

            # The ode is done as a loop over the locations in the initial condition.
            # As locations that don't reimpose mandates produce identical forecasts,
            # subset here to only the locations that reimpose mandates for speed.
            initial_condition_subset = initial_condition.loc[reimposition_date.index]
            logger.info('Running ODE forecast.', context='compute_ode')
            compartments_subset, chis = model.run_ode_forecast(
                initial_condition_subset,
                model_parameters,
            )

            logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
            compartments = (compartments
                            .sort_index()
                            .drop(compartments_subset.index)
                            .append(compartments_subset)
                            .sort_index())
            system_metrics, output_metrics = model.compute_output_metrics(
                indices,
                compartments,
                postprocessing_params,
                model_parameters,
                hospital_parameters,
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

    logger.info('Prepping outputs.', context='transform')
    ode_params, *_ = model_parameters.to_dfs()
    ode_params = pd.concat([ode_params, beta, beta_hat], axis=1)
    outputs = pd.concat([system_metrics, output_metrics.reset_index(level='observed', drop=True),
                         postprocessing_params.correction_factors_df], axis=1)
#    chis = past_chis.loc[indices.past].append(chis.loc[indices.future]).sort_index()

    logger.info('Writing outputs.', context='write')
    data_interface.save_ode_params(ode_params, scenario, draw_id)
    data_interface.save_components(compartments, scenario, draw_id)
    data_interface.save_raw_covariates(covariates, scenario, draw_id)
    data_interface.save_raw_outputs(outputs, scenario, draw_id)
#    data_interface.save_chis(chis, scenario, draw_id)

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

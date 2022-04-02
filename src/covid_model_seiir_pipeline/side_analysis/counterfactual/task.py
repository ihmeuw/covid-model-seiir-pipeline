from pathlib import Path

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.counterfactual.specification import CounterfactualSpecification
from covid_model_seiir_pipeline.pipeline.counterfactual.data import CounterfactualDataInterface
from covid_model_seiir_pipeline.pipeline.forecasting import model as fmodel

from covid_model_seiir_pipeline.pipeline.counterfactual import model

logger = cli_tools.task_performance_logger


def run_counterfactual(counterfactual_version: str, scenario: str, draw_id: int, progress_bar: bool):
    logger.info(f"Initiating SEIIR counterfactual for scenario {scenario}, draw {draw_id}.", context='setup')
    counterfactual_spec = CounterfactualSpecification.from_path(
        Path(counterfactual_version) / static_vars.COUNTERFACTUAL_SPECIFICATION_FILE
    )
    scenario_spec = counterfactual_spec.scenarios[scenario]
    data_interface = CounterfactualDataInterface.from_specification(counterfactual_spec)
    forecast_spec = data_interface.load_forecast_specification()
    forecast_reference_scenario_spec = forecast_spec.scenarios['reference']

    #################
    # Build indices #
    #################
    logger.info('Loading index building data', context='read')
    past_infections = data_interface.load_past_infections(draw_id)
    betas = data_interface.load_counterfactual_beta(scenario_spec.beta, draw_id)

    past_start_dates = past_infections.reset_index().groupby('location_id').date.min()
    minimum_forecast_start_dates = betas.reset_index().groupby('location_id').date.min()
    target_start_date = pd.Series(pd.Timestamp(scenario_spec.start_date), index=past_start_dates.index, name='date')
    forecast_start_dates = np.maximum(target_start_date, minimum_forecast_start_dates)
    forecast_end_dates = betas.reset_index().groupby('location_id').date.max()

    logger.info('Building indices', context='transform')
    indices = fmodel.Indices(
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
    # Vaccine data, of course.
    vaccinations = data_interface.load_vaccinations(scenario_spec.vaccine_coverage)
    # Variant prevalences.
    rhos = data_interface.load_variant_prevalence(scenario_spec.variant_prevalence)

    # Collate all the parameters, ensure consistent index, etc.
    logger.info('Processing inputs into model parameters.', context='transform')
    model_parameters = model.build_model_parameters(
        indices,
        ode_params,
        betas,
        rhos,
        vaccinations,
    )

    # Pull in compartments from the fit and subset out the initial condition.
    logger.info('Loading past compartment data.', context='read')
    past_compartments = data_interface.load_compartments(draw_id=draw_id)
    initial_condition = past_compartments.loc[indices.initial_condition].reset_index(level='date', drop=True)

    ###################################################
    # Construct parameters for postprocessing results #
    ###################################################
    logger.info('Loading results processing input data.', context='read')
    past_deaths = data_interface.load_past_deaths(draw_id=draw_id)
    ratio_data = data_interface.load_ratio_data(draw_id=draw_id)
    hospital_parameters = data_interface.get_hospital_parameters()
    correction_factors = data_interface.load_hospital_correction_factors()

    logger.info('Prepping results processing parameters.', context='transform')
    postprocessing_params = fmodel.build_postprocessing_parameters(
        indices,
        past_compartments,
        past_infections,
        past_deaths,
        ratio_data,
        model_parameters,
        correction_factors,
        hospital_parameters,
        forecast_reference_scenario_spec,
    )

    logger.info('Running ODE forecast.', context='compute_ode')
    future_components = fmodel.run_ode_model(
        initial_condition,
        model_parameters.reindex(indices.future),
        progress_bar,
    )
    logger.info('Processing ODE results and computing deaths and infections.', context='compute_results')
    components, system_metrics, output_metrics = fmodel.compute_output_metrics(
        indices,
        future_components,
        postprocessing_params,
        model_parameters,
        hospital_parameters,
    )

    logger.info('Prepping outputs.', context='transform')
    ode_params = model_parameters.to_df()
    outputs = pd.concat([system_metrics.to_df(), output_metrics.to_df(),
                         postprocessing_params.correction_factors_df], axis=1)

    logger.info('Writing outputs.', context='write')
    data_interface.save_ode_params(ode_params, scenario, draw_id)
    data_interface.save_components(components, scenario, draw_id)
    data_interface.save_raw_outputs(outputs, scenario, draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_counterfactual_version
@cli_tools.with_scenario
@cli_tools.with_draw_id
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def counterfactual(counterfactual_version: str, scenario: str, draw_id: int,
                   progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)

    run = cli_tools.handle_exceptions(run_counterfactual, logger, with_debugger)
    run(counterfactual_version=counterfactual_version,
        scenario=scenario,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    counterfactual()

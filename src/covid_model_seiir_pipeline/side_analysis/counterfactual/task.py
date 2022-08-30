import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.forecasting import model as forecast_model
from covid_model_seiir_pipeline.side_analysis.counterfactual import model
from covid_model_seiir_pipeline.side_analysis.counterfactual.specification import (
    CounterfactualSpecification,
    COUNTERFACTUAL_JOBS,
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.data import (
    CounterfactualDataInterface,
)


logger = cli_tools.task_performance_logger


def run_counterfactual_scenario(counterfactual_version: str, scenario: str, draw_id: int, progress_bar: bool):
    logger.info(f"Initiating SEIIR counterfactual for scenario {scenario}, draw {draw_id}.", context='setup')
    specification = CounterfactualSpecification.from_version_root(counterfactual_version)
    num_cores = specification.workflow.task_specifications[COUNTERFACTUAL_JOBS.counterfactual_scenario].num_cores
    scenario_spec = specification.scenarios[scenario]
    data_interface = CounterfactualDataInterface.from_specification(specification)

    #################
    # Build indices #
    #################
    # The hardest thing to keep consistent is data alignment. We have about 100
    # unique datasets in this model, and they need to be aligned consistently
    # to do computation.
    logger.info('Loading index building data', context='read')
    past_compartments = data_interface.load_past_compartments(
        draw_id=draw_id,
        initial_condition_measure=scenario_spec.initial_condition,
    )
    past_compartments = past_compartments.loc[past_compartments.notnull().any(axis=1)]
    beta = data_interface.load_counterfactual_beta(scenario_spec.beta, draw_id)
    indices = model.build_indices(
        scenario_spec=scenario_spec,
        past_compartments=past_compartments,
        beta=beta,
    )

    ########################################
    # Build parameters for the SEIIR model #
    ########################################
    logger.info('Loading SEIIR parameter input data.', context='read')
    ode_params = data_interface.load_input_ode_params(
        draw_id=draw_id,
        initial_condition_measure=scenario_spec.initial_condition,
    )
    # Vaccine data, of course.
    vaccinations = data_interface.load_vaccine_uptake(scenario_spec.vaccine_coverage)
    etas = data_interface.load_vaccine_risk_reduction(scenario_spec.vaccine_coverage)
    prior_ratios = data_interface.load_rates(
        draw_id=draw_id,
        initial_condition_measure=scenario_spec.initial_condition,
    )
    phis = data_interface.load_phis(draw_id=draw_id)
    hospital_cf = data_interface.load_hospitalizations(measure='correction_factors')
    hospital_parameters = data_interface.get_hospital_params()

    # Collate all the parameters, ensure consistent index, etc.
    logger.info('Processing inputs into model parameters.', context='transform')
    model_parameters = model.build_model_parameters(
        indices=indices,
        counterfactual_beta=beta['beta'],
        ode_parameters=ode_params,
        prior_ratios=prior_ratios,
        vaccinations=vaccinations,
        etas=etas,
        phis=phis,
    )
    hospital_cf = forecast_model.forecast_correction_factors(
        indices=indices,
        correction_factors=hospital_cf,
        hospital_parameters=hospital_parameters,
    )

    initial_condition = past_compartments.loc[indices.past].reindex(indices.full, fill_value=0.)
    initial_condition[initial_condition < 0.] = 0.

    logger.info('Running ODE forecast.', context='compute_ode')
    compartments, chis = forecast_model.run_ode_forecast(
        initial_condition,
        model_parameters,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    system_metrics = forecast_model.compute_output_metrics(
        indices,
        compartments,
        model_parameters,
        ode_params,
        hospital_parameters,
        hospital_cf,
    )

    logger.info('Prepping outputs.', context='transform')
    counterfactual_ode_params = pd.concat([model_parameters.base_parameters, beta], axis=1)
    counterfactual_ode_params['beta_hat'] = np.nan
    for measure in ['death', 'case', 'admission']:
        counterfactual_ode_params[f'exposure_to_{measure}'] = ode_params[f'exposure_to_{measure}'].iloc[0]
    logger.info('Writing outputs.', context='write')
    data_interface.save_ode_params(counterfactual_ode_params, scenario, draw_id)
    data_interface.save_components(compartments, scenario, draw_id)
    data_interface.save_raw_outputs(system_metrics, scenario, draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_counterfactual_version
@cli_tools.with_scenario
@cli_tools.with_draw_id
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def counterfactual_scenario(counterfactual_version: str,
                            scenario: str, draw_id: int,
                            progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)

    run = cli_tools.handle_exceptions(run_counterfactual_scenario, logger, with_debugger)
    run(counterfactual_version=counterfactual_version,
        scenario=scenario,
        draw_id=draw_id,
        progress_bar=progress_bar)


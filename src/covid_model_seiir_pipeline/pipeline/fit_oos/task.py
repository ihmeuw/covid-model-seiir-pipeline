from pathlib import Path

import click

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.fit_oos.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit_oos.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit_oos import model


logger = cli_tools.task_performance_logger


def run_beta_fit(fit_version: str, scenario: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    # Build helper abstractions
    fit_spec_file = Path(fit_version) / static_vars.FIT_SPECIFICATION_FILE
    fit_specification = FitSpecification.from_path(fit_spec_file)
    data_interface = FitDataInterface.from_specification(fit_specification)

    logger.info('Loading ODE fit input data', context='read')
    past_infection_data = data_interface.load_past_infection_data(draw_id=draw_id)
    population = data_interface.load_total_population()
    vaccinations = data_interface.load_vaccine_info('reference')
    variant_prevalence = data_interface.load_variant_prevalence()

    logger.info('Prepping ODE fit parameters.', context='transform')
    infections = model.clean_infection_data_measure(past_infection_data, 'infections')
    fit_params = fit_specification.scenarios[scenario].to_dict()
    ode_parameters = model.prepare_ode_fit_parameters(
        infections,
        population,
        vaccinations,
        variant_prevalence,
        fit_params,
        draw_id,
    )

    logger.info('Running ODE fit', context='compute_ode')
    beta_fit, compartments = model.run_ode_fit(
        ode_parameters=ode_parameters,
        progress_bar=progress_bar,
    )

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_scenario
@cli_tools.with_draw_id
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def beta_fit(fit_version: str, scenario: str, draw_id: int,
             progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_fit, logger, with_debugger)
    run(regression_version=fit_version,
        scenario=scenario,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    beta_fit()

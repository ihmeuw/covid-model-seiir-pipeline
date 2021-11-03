from pathlib import Path

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.pipeline.regression import model
from covid_model_seiir_pipeline.pipeline.parameter_fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.parameter_fit.specification import FitSpecification


logger = cli_tools.task_performance_logger


def run_parameter_fit(fit_version: str, scenario: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    # Build helper abstractions
    fit_spec_file = Path(fit_version) / static_vars.FIT_SPECIFICATION_FILE
    fit_specification = FitSpecification.from_path(fit_spec_file)
    data_interface = FitDataInterface.from_specification(fit_specification)

    logger.info('Loading ODE fit input data', context='read')
    past_infection_data = data_interface.load_past_infection_data(draw_id=draw_id)
    population = data_interface.load_five_year_population()
    rhos = data_interface.load_variant_prevalence()
    vaccinations, boosters = data_interface.load_vaccinations()
    etas = data_interface.load_etas()
    covariates = data_interface.load_covariates(['mask_use'])

    logger.info('Prepping ODE fit parameters.', context='transform')
    infections = model.clean_infection_data_measure(past_infection_data, 'infections')
    fit_params = fit_specification.scenarios[scenario]

    np.random.seed(draw_id)
    sampled_params = model.sample_params(
        infections.index, fit_params.to_dict(),
        params_to_sample=['alpha', 'sigma', 'gamma', 'pi'] + [f'kappa_{v}' for v in VARIANT_NAMES],
        draw_id=draw_id,
    )

    natural_waning_params = (0.8, 270, 0.1, 720)
    natural_waning_matrix = pd.DataFrame(
        data=np.array([
            [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]),
        columns=VARIANT_NAMES,
        index=VARIANT_NAMES,
    )

    phis = model.prepare_phis(
        infections,
        covariates,
        natural_waning_matrix,
        natural_waning_params,
    )

    ode_parameters = model.prepare_ode_fit_parameters(
        infections,
        rhos,
        vaccinations,
        boosters,
        etas,
        phis,
        sampled_params,
    )

    initial_condition = model.make_initial_condition(
        ode_parameters,
        population,
    )

    logger.info('Running ODE fit', context='compute_ode')
    beta, compartments = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        progress_bar=progress_bar,
    )

    ode_parameters, _, etas, phis = ode_parameters.to_dfs()

    data_interface.save_betas(beta, scenario=scenario, draw_id=draw_id)
    data_interface.save_compartments(compartments, scenario=scenario, draw_id=draw_id)
    data_interface.save_ode_parameters(ode_parameters, scenario=scenario, draw_id=draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_scenario
@cli_tools.with_draw_id
@cli_tools.with_progress_bar
@cli_tools.add_verbose_and_with_debugger
def parameter_fit(fit_version: str, scenario: str, draw_id: int,
                  progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_parameter_fit, logger, with_debugger)
    run(fit_version=fit_version,
        scenario=scenario,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    parameter_fit()

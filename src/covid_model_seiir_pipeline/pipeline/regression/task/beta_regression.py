from pathlib import Path

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
    static_vars,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.pipeline.regression import model


logger = cli_tools.task_performance_logger


def run_beta_regression(regression_version: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta regression.', context='setup')
    # Build helper abstractions
    regression_spec_file = Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    regression_specification = RegressionSpecification.from_path(regression_spec_file)
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    logger.info('Loading ODE fit input data', context='read')
    hierarchy = data_interface.load_hierarchy()
    past_infection_data = data_interface.load_past_infection_data(draw_id=draw_id)
    population = data_interface.load_five_year_population()
    rhos = data_interface.load_variant_prevalence()
    vaccinations, boosters = data_interface.load_vaccinations()

    logger.info('Prepping ODE fit parameters.', context='transform')
    infections = model.clean_infection_data_measure(past_infection_data, 'infections')
    covariates = data_interface.load_covariates(list(regression_specification.covariates))
    regression_params = regression_specification.regression_parameters.to_dict()

    np.random.seed(draw_id)
    sampled_params = model.sample_params(
        infections.index, regression_params,
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

    vaccine_waning_params = (0.5, 180, 0.1, 720)
    booster_waning_params = (0.5, 180, 0.1, 720)

    etas_immune, etas_protected, total_vaccinations = model.prepare_etas_and_vaccinations(
        infections,
        vaccinations,
        vaccine_waning_params,
    )

    etas_booster_immune, etas_booster_protected, total_boosters = model.prepare_etas_and_vaccinations(
        infections,
        boosters,
        booster_waning_params,
    )

    ode_parameters = model.prepare_ode_fit_parameters(
        infections,
        rhos,
        total_vaccinations,
        total_boosters,
        etas_immune,
        etas_booster_immune,
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
    )

    logger.info('Loading regression input data', context='read')
    
    gaussian_priors = data_interface.load_priors(regression_specification.covariates.values())
    prior_coefficients = data_interface.load_prior_run_coefficients(draw_id=draw_id)
    if gaussian_priors and prior_coefficients:
        raise NotImplementedError

    logger.info('Fitting beta regression', context='compute_regression')
    coefficients = model.run_beta_regression(
        beta,
        covariates,
        regression_specification.covariates.values(),
        gaussian_priors,
        prior_coefficients,
        hierarchy,
    )
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_hat')

    # Format and save data.
    logger.info('Prepping outputs', context='transform')
    betas = pd.concat([beta, beta_hat], axis=1).reindex(infections.index)
    deaths = model.clean_infection_data_measure(past_infection_data, 'deaths')
    ode_parameters, _, etas, phis = ode_parameters.to_dfs()

    logger.info('Writing outputs', context='write')
    data_interface.save_infections(infections, draw_id=draw_id)
    data_interface.save_deaths(deaths, draw_id=draw_id)
    data_interface.save_betas(betas, draw_id=draw_id)
    data_interface.save_compartments(compartments, draw_id=draw_id)
    data_interface.save_coefficients(coefficients, draw_id=draw_id)
    data_interface.save_ode_parameters(ode_parameters, draw_id=draw_id)
    data_interface.save_etas(etas, draw_id=draw_id)
    data_interface.save_phis(phis, draw_id=draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_regression_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_regression(regression_version: str, draw_id: int,
                    progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_regression, logger, with_debugger)
    run(regression_version=regression_version,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    beta_regression()

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_beta_fit(fit_version: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)

    logger.info('Loading rates data', context='read')
    hierarchy = data_interface.load_modeling_hierarchy().reset_index()

    logger.info('Running first-pass rates model', context='rates_model')
    rates, epi_measures, lags = model.run_rates_model(hierarchy)

    logger.info('Loading ODE fit input data', context='read')

    risk_group_pops = data_interface.load_population(measure='risk_group')
    rhos = data_interface.load_variant_prevalence(scenario='reference')
    vaccinations = data_interface.load_vaccine_uptake(scenario='reference')
    etas = data_interface.load_vaccine_risk_reduction(scenario='reference')
    natural_waning_dist = data_interface.load_waning_parameters(measure='natural_waning_distribution').set_index(
        ['endpoint', 'days'])
    phi = pd.DataFrame(
        data=np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]),
        columns=VARIANT_NAMES,
        index=pd.Index(VARIANT_NAMES, name='variant'),
    )

    logger.info('Prepping ODE fit parameters.', context='transform')
    regression_params = specification.fit_parameters.to_dict()



    ode_parameters = model.prepare_ode_fit_parameters(
        rates,
        epi_measures,
        rhos,
        vaccinations,
        etas,
        natural_waning_dist,
        phi,
        regression_params,
        draw_id,
    )

    initial_condition = model.make_initial_condition(
        ode_parameters,
        population,
    )

    logger.info('Running ODE fit', context='compute_ode')
    beta, chis, compartments = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        progress_bar=progress_bar,
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
    ode_parameters, _, etas, _, phis = ode_parameters.to_dfs()

    logger.info('Writing outputs', context='write')
    data_interface.save_infections(infections, draw_id=draw_id)
    data_interface.save_deaths(deaths, draw_id=draw_id)
    data_interface.save_betas(betas, draw_id=draw_id)
    data_interface.save_compartments(compartments, draw_id=draw_id)
    data_interface.save_coefficients(coefficients, draw_id=draw_id)
    data_interface.save_ode_parameters(ode_parameters, draw_id=draw_id)
    data_interface.save_chis(chis, draw_id=draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_fit(fit_version: str, draw_id: int,
          progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_fit, logger, with_debugger)
    run(fit_version=fit_version,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    beta_fit()

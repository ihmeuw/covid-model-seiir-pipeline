from pathlib import Path

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    math,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    clean_infection_data_measure,
    prepare_ode_fit_parameters,
    run_ode_fit,
    sample_params,
    reslime,
)
from covid_model_seiir_pipeline.pipeline.fit_oos.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit_oos.specification import FitSpecification


logger = cli_tools.task_performance_logger


def run_beta_fit(fit_version: str, scenario: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    # Build helper abstractions
    fit_spec_file = Path(fit_version) / static_vars.FIT_SPECIFICATION_FILE
    fit_specification = FitSpecification.from_path(fit_spec_file)
    data_interface = FitDataInterface.from_specification(fit_specification)

    logger.info('Loading ODE fit input data', context='read')
    hierarchy = data_interface.load_hierarchy()
    past_infection_data = data_interface.load_past_infection_data(draw_id=draw_id)
    population = data_interface.load_five_year_population()
    rhos = data_interface.load_variant_prevalence()
    vaccinations = data_interface.load_vaccinations()

    logger.info('Prepping ODE fit parameters.', context='transform')
    infections = clean_infection_data_measure(past_infection_data, 'infections')
    fit_params = fit_specification.scenarios[scenario]

    np.random.seed(draw_id)
    sampled_params = sample_params(
        infections.index, fit_params.to_dict(),
        params_to_sample=['alpha', 'sigma', 'gamma1', 'gamma2', 'kappa', 'phi', 'psi', 'pi', 'chi']
    )
    ode_parameters = prepare_ode_fit_parameters(
        infections,
        population,
        rhos,
        vaccinations,
        sampled_params,
    )

    logger.info('Running ODE fit', context='compute_ode')
    beta, compartments = run_ode_fit(
        ode_parameters=ode_parameters,
        progress_bar=progress_bar,
    )

    covariates = data_interface.load_covariates([
        'pneumonia',
        'mobility',
        'mask_use',
        'testing',
        'air_pollution_pm_2_5',
        'smoking_prevalence',
        'lri_mortality',
        'proportion_under_100m',
        'proportion_over_2_5k',
    ])
    prior_coefficients = data_interface.load_prior_run_coefficients(draw_id=draw_id)
    log_beta_hat = math.compute_beta_hat(covariates, prior_coefficients)
    log_beta_residual = (np.log(beta['beta_wild']) - log_beta_hat).rename('log_beta_residual')

    today = pd.Timestamp(fit_params.max_date)
    b117_only = rhos[(rhos['rho'] > 0) & (rhos['rho_variant'] == 0)].reset_index()
    b117_only = b117_only[(b117_only.date < today)]
    max_rho = b117_only.groupby('location_id').rho.max()
    if fit_params.location_filter:
        locs_to_fit = fit_params.location_filter
    elif fit_params.threshold:
        locs_to_fit = max_rho[max_rho > fit_params.threshold].index.tolist()
    regression_index = b117_only[b117_only.location_id.isin(locs_to_fit)].set_index(['location_id', 'date']).index

    regression_inputs = pd.merge(log_beta_residual.dropna(), rhos.loc[regression_index, ['rho']],
                                 how='inner',
                                 on=log_beta_residual.index.names)
    group_cols = ['super_region_id', 'region_id', 'location_id']
    regression_inputs = (regression_inputs
                         .merge(hierarchy[group_cols], on='location_id')
                         .reset_index()
                         .set_index(group_cols)
                         .sort_index())
    regression_inputs['intercept'] = 1.0

    intercept_model = reslime.PredictorModel(
        'intercept',
        group_level='location_id',
    )
    rho_model = reslime.PredictorModel(
        'rho',
        bounds=(0.0, np.inf),
    )
    predictor_set = reslime.PredictorModelSet([intercept_model, rho_model])
    mr_data = reslime.MRData(
        data=regression_inputs.reset_index(),
        response_column='log_beta_residual',
        predictors=[p.name for p in predictor_set],
        group_columns=group_cols,
    )
    mr_model = reslime.MRModel(mr_data, predictor_set)
    coefficients = mr_model.fit_model().reset_index(level=['super_region_id', 'region_id'], drop=True)

    data_interface.save_betas(beta, scenario=scenario, draw_id=draw_id)
    data_interface.save_compartments(compartments, scenario=scenario, draw_id=draw_id)
    data_interface.save_ode_parameters(ode_parameters.to_df(), scenario=scenario, draw_id=draw_id)
    data_interface.save_regression_params(regression_inputs, scenario=scenario, draw_id=draw_id)
    data_interface.save_coefficients(coefficients, scenario=scenario, draw_id=draw_id)

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
    run(fit_version=fit_version,
        scenario=scenario,
        draw_id=draw_id,
        progress_bar=progress_bar)


if __name__ == '__main__':
    beta_fit()

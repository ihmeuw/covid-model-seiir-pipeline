import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_beta_fit(fit_version: str, draw_id: int, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_threads = specification.workflow.task_specifications[FIT_JOBS.beta_fit].num_cores

    logger.info('Loading beta fit data', context='read')
    mr_hierarchy = data_interface.load_hierarchy(name='mr')
    pred_hierarchy = data_interface.load_hierarchy(name='pred')
    total_population = data_interface.load_population(measure='total').population
    five_year_population = data_interface.load_population(measure='five_year').population
    risk_group_population = data_interface.load_population(measure='risk_group')
    epi_measures = data_interface.load_reported_epi_data()
    mortality_scalar = data_interface.load_total_covid_scalars(draw_id)['scalar']
    age_patterns = data_interface.load_age_patterns()
    seroprevalence = data_interface.load_seroprevalence(draw_id=draw_id).reset_index()
    sensitivity_data = data_interface.load_sensitivity(draw_id=draw_id)
    testing_capacity = data_interface.load_testing_data()['testing_capacity']
    covariate_pool = data_interface.load_covariate_options(draw_id=draw_id)
    rhos = data_interface.load_variant_prevalence(scenario='reference')
    variant_prevalence = rhos.drop(columns='ancestral').sum(axis=1)
    vaccinations = data_interface.load_vaccine_uptake(scenario='reference')
    etas = data_interface.load_vaccine_risk_reduction(scenario='reference')
    natural_waning_dist = data_interface.load_waning_parameters(measure='natural_waning_distribution').set_index('days')
    natural_immunity_matrix = pd.DataFrame(
        data=np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]),
        columns=VARIANT_NAMES,
        index=pd.Index(VARIANT_NAMES, name='variant'),
    )
    mr_covariates = []
    for covariate in model.COVARIATE_POOL:
        cov = (data_interface
               .load_covariate(covariate, 'reference')
               .groupby('location_id')[f'{covariate}_reference']
               .mean().rename(covariate))
        mr_covariates.append(cov)

    logger.info('Sampling rates parameters', context='transform')
    durations = model.sample_durations(specification.rates_parameters, draw_id)
    variant_severity = model.sample_variant_severity(specification.rates_parameters, draw_id)
    day_inflection = model.sample_day_inflection(specification.rates_parameters, draw_id)

    logger.info('Rescaling deaths and formatting epi measures', context='transform')
    epi_measures = model.format_epi_measures(epi_measures, mr_hierarchy, pred_hierarchy, mortality_scalar, durations)

    logger.info('Subsetting seroprevalence for first pass rates model', context='transform')
    first_pass_seroprevalence = model.subset_seroprevalence(
        seroprevalence=seroprevalence,
        epi_data=epi_measures,
        variant_prevalence=variant_prevalence,
        population=total_population,
        params=specification.rates_parameters,
    )

    logger.info('Generating naive infections for first pass rates model', context='transform')
    daily_deaths = epi_measures['smoothed_daily_deaths'].dropna()
    naive_ifr = specification.rates_parameters.naive_ifr
    daily_infections = (daily_deaths / naive_ifr).rename('daily_infections').reset_index()
    daily_infections['date'] -= pd.Timedelta(days=durations.exposure_to_death)
    daily_infections = daily_infections.set_index(['location_id', 'date']).loc[:, 'daily_infections']

    logger.info('Running first-pass rates model', context='rates_model_1')
    first_pass_rates, first_pass_rates_data = model.run_rates_pipeline(
        epi_data=epi_measures,
        age_patterns=age_patterns,
        seroprevalence=first_pass_seroprevalence,
        covariates=mr_covariates,
        covariate_pool=covariate_pool,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        total_population=total_population,
        age_specific_population=five_year_population,
        testing_capacity=testing_capacity,
        variant_prevalence=variant_prevalence,
        daily_infections=daily_infections,
        durations=durations,
        variant_rrs=variant_severity,
        params=specification.rates_parameters,
        day_inflection=day_inflection,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )

    logger.info('Prepping ODE fit parameters.', context='transform')
    ode_parameters, first_pass_base_rates = model.prepare_ode_fit_parameters(
        rates=first_pass_rates,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_immunity_matrix,
        variant_severity=variant_severity,
        fit_params=specification.fit_parameters,
        hierarchy=pred_hierarchy,
        draw_id=draw_id,
    )

    logger.info('Building initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        ode_parameters,
        first_pass_base_rates,
        risk_group_population,
    )

    logger.info('Running ODE fit', context='compute_ode')
    compartments, first_pass_betas, chis = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=specification.workflow.task_specifications['beta_fit'].num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Prepping first pass ODE outputs for second pass rates model', context='transform')
    first_pass_posterior_epi_measures, pct_unvaccinated = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations
    )

    hospitalized_weights = model.get_all_age_rate(
        rate_age_pattern=age_patterns['ihr'],
        weight_age_pattern=age_patterns['seroprevalence'],
        age_spec_population=five_year_population,
    )
    sensitivity, adjusted_seroprevalence = model.apply_sensitivity_adjustment(
        sensitivity_data=sensitivity_data,
        hospitalized_weights=hospitalized_weights,
        seroprevalence=seroprevalence,
        daily_infections=first_pass_posterior_epi_measures['daily_total_infections'].rename('daily_infections'),
        durations=durations,
    )
    adjusted_seroprevalence = adjusted_seroprevalence.merge(pct_unvaccinated, how='left')
    if adjusted_seroprevalence['pct_unvaccinated'].isnull().any():
        logger.warning('Unmatched sero-survey dates')
    adjusted_seroprevalence['seroprevalence'] *= adjusted_seroprevalence['pct_unvaccinated']
    del adjusted_seroprevalence['pct_unvaccinated']

    logger.info('Running second-pass rates model', context='rates_model_2')
    second_pass_rates, second_pass_rates_data = model.run_rates_pipeline(
        epi_data=first_pass_posterior_epi_measures,
        age_patterns=age_patterns,
        seroprevalence=adjusted_seroprevalence,
        covariates=mr_covariates,
        covariate_pool=covariate_pool,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        total_population=total_population,
        age_specific_population=five_year_population,
        testing_capacity=testing_capacity,
        variant_prevalence=variant_prevalence,
        daily_infections=daily_infections,
        durations=durations,
        variant_rrs=variant_severity,
        params=specification.rates_parameters,
        day_inflection=day_inflection,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )

    logger.info('Prepping ODE fit parameters for second pass model.', context='transform')
    ode_parameters, second_pass_base_rates = model.prepare_ode_fit_parameters(
        rates=second_pass_rates,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_immunity_matrix,
        variant_severity=variant_severity,
        fit_params=specification.fit_parameters,
        hierarchy=pred_hierarchy,
        draw_id=draw_id,
    )

    logger.info('Rebuilding initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        ode_parameters,
        second_pass_base_rates,
        risk_group_population,
    )

    logger.info('Running second pass ODE fit', context='compute_ode')
    compartments, second_pass_betas, chis = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=specification.workflow.task_specifications['beta_fit'].num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Prepping outputs.', context='transform')

    base_parameters = ode_parameters.to_dict()['base_parameters']
    out_params = []
    keep = ['alpha', 'sigma', 'gamma', 'pi', 'kappa', 'weight']
    for param in keep:
        out_params.append(base_parameters.filter(like=param).iloc[0])
    out_params = pd.concat(out_params)
    for name, duration in durations._asdict().items():
        out_params.loc[name] = duration
    out_params = out_params.reset_index()
    out_params.columns = ['parameter', 'value']

    rates_data = []
    for round_id, dataset in enumerate([first_pass_rates_data, second_pass_rates_data]):
        for measure in ['ifr', 'ihr', 'idr']:
            df = dataset._asdict()[measure]
            df = (df
                  .loc[:, ['location_id', 'mean_infection_date', 'data_id', measure]]
                  .rename(columns={measure: 'value', 'mean_infection_date': 'date'}))
            df['measure'] = measure
            df['round'] = round_id + 1
            rates_data.append(df)
    rates_data = pd.concat(rates_data)

    first_pass_rates = pd.concat([r.drop(columns='lag') for r in first_pass_rates], axis=1)
    first_pass_rates.loc[:, 'round'] = 1
    second_pass_rates = pd.concat([r.drop(columns='lag') for r in second_pass_rates], axis=1)
    second_pass_rates.loc[:, 'round'] = 2
    prior_rates = pd.concat([first_pass_rates, second_pass_rates])

    first_pass_betas.loc[:, 'round'] = 1
    second_pass_betas.loc[:, 'round'] = 2
    betas = pd.concat([first_pass_betas, second_pass_betas])

    second_pass_posterior_epi_measures, _ = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations
    )
    first_pass_posterior_epi_measures.loc[:, 'round'] = 1
    second_pass_posterior_epi_measures.loc[:, 'round'] = 2
    posterior_epi_measures = pd.concat([first_pass_posterior_epi_measures, second_pass_posterior_epi_measures])

    idx_cols = ['data_id', 'location_id', 'date', 'is_outlier']
    out_seroprevalence = seroprevalence.loc[:, idx_cols + ['reported_seroprevalence', 'seroprevalence']]
    adjusted_seroprevalence = (adjusted_seroprevalence
                               .loc[:, idx_cols + ['seroprevalence']]
                               .rename(columns={'seroprevalence': 'adjusted_seroprevalence'}))
    out_seroprevalence = out_seroprevalence.merge(adjusted_seroprevalence)

    logger.info('Writing outputs', context='write')
    data_interface.save_ode_params(out_params, draw_id=draw_id)
    data_interface.save_phis(ode_parameters.phis, draw_id=draw_id)
    data_interface.save_input_epi_measures(epi_measures, draw_id=draw_id)
    data_interface.save_rates(prior_rates, draw_id=draw_id)
    data_interface.save_rates_data(rates_data, draw_id=draw_id)
    data_interface.save_posterior_epi_measures(posterior_epi_measures, draw_id=draw_id)
    data_interface.save_compartments(compartments, draw_id=draw_id)
    data_interface.save_fit_beta(betas, draw_id=draw_id)
    data_interface.save_final_seroprevalence(out_seroprevalence, draw_id=draw_id)

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

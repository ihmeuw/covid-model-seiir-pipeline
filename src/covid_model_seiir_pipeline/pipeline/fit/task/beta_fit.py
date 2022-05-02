import click
import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_beta_fit(fit_version: str, measure: str, draw_id: int, progress_bar: bool) -> None:
    logger.info(f'Starting beta fit for measure {measure} draw {draw_id}.', context='setup')
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
    testing_capacity_offset = 1
    testing_capacity += testing_capacity_offset
    covariate_pool = data_interface.load_covariate_options(draw_id=draw_id)
    rhos = data_interface.load_variant_prevalence(scenario='reference')
    variant_prevalence = rhos.drop(columns='ancestral').sum(axis=1)
    vaccinations = data_interface.load_vaccine_uptake(scenario='reference')
    etas = data_interface.load_vaccine_risk_reduction(scenario='reference')
    natural_waning_dist = data_interface.load_waning_parameters(measure='natural_waning_distribution').set_index('days')
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
    sampled_ode_params, natural_waning_matrix = model.sample_ode_params(
        variant_severity, specification.fit_parameters, draw_id
    )

    logger.info('Rescaling deaths and formatting epi measures', context='transform')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## MANUAL DROPS ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    drop_location_ids = {
        # if it's a parent location, will apply to all children as well
        'death': [44533],
        'case': [],
        'admission':[],
    }
    drop_location_ids = drop_location_ids[measure]
    drop_location_ids = [(pred_hierarchy.loc[pred_hierarchy['path_to_top_parent']
                                             .apply(lambda x: str(loc_id) in x.split(',')), 'location_id'].to_list())
                         for loc_id in drop_location_ids]
    if drop_location_ids:
        drop_location_ids = np.unique((np.hstack(drop_location_ids))).tolist()
        drop_location_names = (pred_hierarchy
                               .set_index('location_id')
                               .loc[drop_location_ids, 'location_name']
                               .to_list())
        drop_locations = [f'{drop_location_name} ({drop_location_id})'
                          for drop_location_id, drop_location_name in zip(drop_location_ids, drop_location_names)]
        logger.warning(f'Dropping data for the following locations:\n'
                       f"{'; '.join(drop_locations)}")
        epi_measures = epi_measures.drop(drop_location_ids)
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    epi_measures = model.format_epi_measures(epi_measures, mr_hierarchy, pred_hierarchy, mortality_scalar, durations)
    epi_measures = model.enforce_epi_threshold(epi_measures, measure, mortality_scalar)

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
    init_daily_infections = (daily_deaths / naive_ifr).rename(
        'daily_infections').reset_index()
    init_daily_infections['date'] -= pd.Timedelta(days=durations.exposure_to_death)
    init_daily_infections = init_daily_infections.set_index(['location_id', 'date']).loc[:,
                            'daily_infections']

    logger.info('Running first-pass rates model', context='rates_model_1')
    first_pass_rates, first_pass_rates_data = model.run_rates_pipeline(
        measure=measure,
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
        daily_infections=init_daily_infections,
        durations=durations,
        variant_rrs=variant_severity,
        params=specification.rates_parameters,
        day_inflection=day_inflection,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )

    logger.info('Prepping ODE fit parameters.', context='transform')
    ode_parameters, first_pass_base_rates = model.prepare_ode_fit_parameters(
        measure=measure,
        rates=first_pass_rates,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_waning_matrix,
        sampled_ode_params=sampled_ode_params,
        hierarchy=pred_hierarchy,
        population=total_population,
        draw_id=draw_id,
    )

    logger.info('Building initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        measure,
        ode_parameters,
        first_pass_base_rates,
        risk_group_population,
    )

    logger.info('Running ODE fit', context='compute_ode')
    first_pass_compartments, first_pass_betas, chis = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=specification.workflow.task_specifications['beta_fit'].num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Prepping first pass ODE outputs for second pass rates model',
                context='transform')
    first_pass_posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=first_pass_compartments,
        durations=durations
    )
    agg_first_pass_posterior_epi_measures = model.aggregate_posterior_epi_measures(
        measure=measure,
        epi_measures=epi_measures,
        posterior_epi_measures=first_pass_posterior_epi_measures,
        hierarchy=mr_hierarchy
    )

    # Apply location specific adjustments for locations where the model breaks.
    sampled_ode_params = model.rescale_kappas(
        measure,
        sampled_ode_params,
        first_pass_compartments,
        specification.rates_parameters,
        pred_hierarchy,
        draw_id
    )

    omega_severity = specification.rates_parameters.omega_severity_parameterization
    severity_calc = {
        'delta': lambda m: sampled_ode_params[f'kappa_delta_{m}'],
        'omicron': lambda m: sampled_ode_params[f'kappa_omicron_{m}'],
        'average': lambda m: 1 / 2 * (sampled_ode_params[f'kappa_delta_{m}'] + sampled_ode_params[f'kappa_omicron_{m}']),
    }[omega_severity]
    for m in ['case', 'admission', 'death']:
        sampled_ode_params[f'kappa_omega_{m}'] = severity_calc(m)

    pct_unvaccinated = (
        (agg_first_pass_posterior_epi_measures['cumulative_naive_unvaccinated_infections']
         / agg_first_pass_posterior_epi_measures['cumulative_naive_infections'])
            .clip(0, 1)
            .fillna(1)
            .rename('pct_unvaccinated')
            .reset_index()
    )
    pct_unvaccinated['date'] += pd.Timedelta(days=durations.exposure_to_seroconversion)

    hospitalized_weights = model.get_all_age_rate(
        rate_age_pattern=age_patterns['ihr'],
        weight_age_pattern=age_patterns['seroprevalence'],
        age_spec_population=five_year_population,
    )
    sensitivity, adjusted_seroprevalence = model.apply_sensitivity_adjustment(
        sensitivity_data=sensitivity_data,
        hospitalized_weights=hospitalized_weights,
        seroprevalence=seroprevalence,
        daily_infections=(agg_first_pass_posterior_epi_measures['daily_total_infections']
                          .rename('daily_infections')),
        population=total_population,
        durations=durations,
    )
    adjusted_seroprevalence = adjusted_seroprevalence.merge(pct_unvaccinated, how='left')
    if adjusted_seroprevalence['pct_unvaccinated'].isnull().any():
        logger.warning('Unmatched sero-survey dates')
    adjusted_seroprevalence['seroprevalence'] *= adjusted_seroprevalence['pct_unvaccinated']

    logger.info('Running second-pass rates model', context='rates_model_2')
    second_pass_rates, second_pass_rates_data = model.run_rates_pipeline(
        measure=measure,
        epi_data=agg_first_pass_posterior_epi_measures,
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
        daily_infections=(
            agg_first_pass_posterior_epi_measures['daily_naive_unvaccinated_infections']
            .rename('daily_infections')),
        durations=durations,
        variant_rrs=variant_severity,
        params=specification.rates_parameters,
        day_inflection=day_inflection,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )

    logger.info('Prepping ODE fit parameters for second pass model.', context='transform')
    ode_parameters, second_pass_base_rates = model.prepare_ode_fit_parameters(
        measure=measure,
        rates=second_pass_rates,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_waning_matrix,
        sampled_ode_params=sampled_ode_params,
        hierarchy=pred_hierarchy,
        population=total_population,
        draw_id=draw_id,
    )

    logger.info('Rebuilding initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        measure,
        ode_parameters,
        second_pass_base_rates,
        risk_group_population,
    )

    logger.info('Running second pass ODE fit', context='compute_ode')
    second_pass_compartments, second_pass_betas, chis = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=specification.workflow.task_specifications['beta_fit'].num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Prepping outputs.', context='transform')
    out_params = ode_parameters.to_dict()['base_parameters']
    for name, duration in durations._asdict().items():
        out_params.loc[:, name] = duration

    rate = {'case': 'idr', 'death': 'ifr', 'admission': 'ihr'}[measure]
    rates_data = []
    for round_id, dataset in enumerate([first_pass_rates_data, second_pass_rates_data]):
        df = (dataset
              .loc[:, ['location_id', 'mean_infection_date', 'data_id', rate]]
              .rename(columns={measure: 'value', 'mean_infection_date': 'date'}))
        df['measure'] = measure
        df['round'] = round_id + 1
        rates_data.append(df)
    rates_data = pd.concat(rates_data)

    first_pass_rates = first_pass_rates.drop(columns='lag')
    first_pass_rates.loc[:, 'round'] = 1
    second_pass_rates = second_pass_rates.drop(columns='lag')
    second_pass_rates.loc[:, 'round'] = 2
    prior_rates = pd.concat([first_pass_rates, second_pass_rates])

    first_pass_betas.loc[:, 'round'] = 1
    second_pass_betas.loc[:, 'round'] = 2
    betas = pd.concat([first_pass_betas, second_pass_betas])

    second_pass_posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=second_pass_compartments,
        durations=durations
    )
    first_pass_posterior_epi_measures.loc[:, 'round'] = 1
    second_pass_posterior_epi_measures.loc[:, 'round'] = 2
    posterior_epi_measures = pd.concat([first_pass_posterior_epi_measures, second_pass_posterior_epi_measures])

    idx_cols = ['data_id', 'location_id', 'date', 'is_outlier']
    out_seroprevalence = seroprevalence.loc[:, idx_cols + ['reported_seroprevalence', 'seroprevalence']]
    adjusted_seroprevalence['seroprevalence'] = (adjusted_seroprevalence['seroprevalence']
                                                 / adjusted_seroprevalence['pct_unvaccinated'])
    adjusted_seroprevalence = (adjusted_seroprevalence
                               .loc[:, idx_cols + ['seroprevalence']]
                               .rename(columns={'seroprevalence': 'adjusted_seroprevalence'}))
    out_seroprevalence = out_seroprevalence.merge(adjusted_seroprevalence)
    out_seroprevalence['sero_date'] = out_seroprevalence['date']
    out_seroprevalence['date'] -= pd.Timedelta(days=durations.exposure_to_seroconversion)

    idr_parameters = model.sample_idr_parameters(specification.rates_parameters, draw_id)
    keep_compartments = [
        'EffectiveSusceptible_all_omicron_all_lr',
        'EffectiveSusceptible_all_omicron_all_hr',
        'Infection_all_delta_all_lr',
        'Infection_all_delta_all_hr',
        'Case_all_delta_all_lr',
        'Case_all_delta_all_hr',
        'Infection_all_all_all_lr',
        'Infection_all_all_all_hr',
        'Case_all_all_all_lr',
        'Case_all_all_all_hr',
    ]
    first_pass_compartments = first_pass_compartments.loc[:, keep_compartments]
    first_pass_compartments['round'] = 1
    second_pass_compartments = second_pass_compartments.loc[:, keep_compartments]
    second_pass_compartments['round'] = 2
    compartments = pd.concat([
        first_pass_compartments,
        second_pass_compartments,
    ])
    for k, v in idr_parameters.items():
        compartments[k] = v

    logger.info('Writing outputs', context='write')
    data_interface.save_ode_params(out_params, measure_version=measure, draw_id=draw_id)
    data_interface.save_input_epi_measures(epi_measures, measure_version=measure, draw_id=draw_id)
    data_interface.save_rates(prior_rates, measure_version=measure, draw_id=draw_id)
    data_interface.save_rates_data(rates_data, measure_version=measure, draw_id=draw_id)
    data_interface.save_posterior_epi_measures(posterior_epi_measures, measure_version=measure, draw_id=draw_id)
    data_interface.save_fit_beta(betas, measure_version=measure, draw_id=draw_id)
    data_interface.save_final_seroprevalence(out_seroprevalence, measure_version=measure, draw_id=draw_id)
    data_interface.save_compartments(compartments, measure_version=measure, draw_id=draw_id)

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_measure
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_fit(fit_version: str, measure: str, draw_id: int,
             progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_fit, logger, with_debugger)
    run(fit_version=fit_version,
        measure=measure,
        draw_id=draw_id,
        progress_bar=progress_bar)

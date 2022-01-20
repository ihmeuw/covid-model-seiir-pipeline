import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
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
    init_daily_infections = (daily_deaths / naive_ifr).rename('daily_infections').reset_index()
    init_daily_infections['date'] -= pd.Timedelta(days=durations.exposure_to_death)
    init_daily_infections = init_daily_infections.set_index(['location_id', 'date']).loc[:, 'daily_infections']

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
        rates=first_pass_rates,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_waning_matrix,
        sampled_ode_params=sampled_ode_params,
        measure_downweights=specification.measure_downweights.to_dict(),
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
    first_pass_posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations
    )
    agg_first_pass_posterior_epi_measures = model.aggregate_posterior_epi_measures(
        epi_measures=epi_measures,
        posterior_epi_measures=first_pass_posterior_epi_measures,
        hierarchy=mr_hierarchy
    )

    delta_infections = compartments.filter(like='Infection_all_delta_all').sum(axis=1).groupby('location_id').max()
    delta_cases = compartments.filter(like='Case_all_delta_all').sum(axis=1).groupby('location_id').max()
    all_infections = compartments.filter(like='Infection_all_all_all').sum(axis=1).groupby('location_id').max()
    all_cases = compartments.filter(like='Case_all_all_all').sum(axis=1).groupby('location_id').max()
    max_idr = 0.9
    p_symptomatic_pre_omicron = 0.5
    p_symptomatic_post_omicron = 1 - model.sample_parameter('p_asymptomatic_omicron', draw_id=draw_id,
                                                            lower=0.85, upper=0.95)
    minimum_asymptomatic_idr_fraction = 0.1
    maximum_asymptomatic_idr = 0.2

    idr_scaling_factors = [
        # 0.2
        (  105, 0.2),  # Antigua and Barbuda
        (  117, 0.2),  # Saint Vincent and the Grenadines
        (  156, 0.2),  # United Arab Emirates
        (   26, 0.2),  # Papua New Guinea
        (  213, 0.2),  # Niger
        (  214, 0.2),  # Nigeria
        (  217, 0.2),  # Sierra Leone
        # 0.4
        (  106, 0.4),  # Bahamas
        (  393, 0.4),  # Saint Kitts and Nevis
        (  376, 0.4),  # Northern Mariana Islands
        (  172, 0.4),  # Equatorial Guinea
        (  173, 0.4),  # Gabon
        (  175, 0.4),  # Burundi
        (  185, 0.4),  # Rwanda
        (  207, 0.4),  # Ghana
        (  210, 0.4),  # Liberia
        # 0.6
        (  170, 0.6),  # Congo
        (  187, 0.6),  # Somalia
        (  190, 0.6),  # Uganda
        (  206, 0.6),  # Gambia
        (  208, 0.6),  # Guinea
        (  216, 0.6),  # Senegal
        (  218, 0.6),  # Togo
        # 0.8
        (  179, 0.8),  # Ethiopia
        (  205, 0.8),  # Côte d'Ivoire
        # 1.2
        (  191, 1.2),  # Zambia
        (  193, 1.2),  # Botswana
        (  201, 1.2),  # Burkina Faso
        # 1.4
        (   49, 1.4),  # North Macedonia
        (  528, 1.4),  # Colorado
        (  529, 1.4),  # Connecticut
        (  531, 1.4),  # District of Columbia
        (  532, 1.4),  # Florida
        (  536, 1.4),  # Illinois
        (  543, 1.4),  # Maryland
        (  553, 1.4),  # New Jersey
        (  555, 1.4),  # New York
        (  558, 1.4),  # Ohio
        (  180, 1.4),  # Kenya
        (  182, 1.4),  # Malawi
        (  198, 1.4),  # Zimbabwe
        # 1.6
        (  396, 1.6),  # San Marino
        (60358, 1.6),  # Aragon
        (60374, 1.6),  # Basque Country
        (60364, 1.6),  # Canary Islands
        (60359, 1.6),  # Cantabria
        (60360, 1.6),  # Castilla-La Mancha
        (60361, 1.6),  # Community of Madrid
        (60363, 1.6),  # Balearic Islands
        (60367, 1.6),  # Castile and León
        (60366, 1.6),  # Murcia
        (  168, 1.6),  # Angola
        # 2.0
        (43860, 2.0),  # Manitoba
        (   97, 2.0),  # Argentina
        (  121, 2.0),  # Bolivia
        (  197, 2.0),  # Eswatini
        # 3.0
        (   83, 3.0),  # Iceland
        (60370, 3.0),  # Navarre
        (60373, 3.0),  # Melilla
        (60376, 3.0),  # La Rioja
        ( 4849, 3.0),  # Delhi
        ( 4863, 3.0),  # Mizoram
        (  196, 3.0),  # South Africa
        (  203, 3.0),  # Cabo Verde
        # 5.0
        (   50, 5.0),  # Montenegro
        (  176, 5.0),  # Comoros
        (  186, 5.0),  # Seychelles
        (  215, 5.0),  # Sao Tome and Principe
    ]
    # IDR = p_s * IDR_s + p_a * IDR_a
    # IDR_a = (IDR - IDR_s * p_s) / p_a
    # IDR_a >= min_frac_a * IDR
    # IDR_a <= 0.2 [post]
    delta_idr = delta_cases / delta_infections
    delta_idr = delta_idr.fillna(all_cases / all_infections)
    capped_delta_idr = np.minimum(delta_idr, max_idr)
    for location_id, idr_scaling_factor in idr_scaling_factors:
        capped_delta_idr.loc[location_id] *= idr_scaling_factor
    idr_asymptomatic = (capped_delta_idr - max_idr * p_symptomatic_pre_omicron) / (1 - p_symptomatic_pre_omicron)
    idr_asymptomatic = np.maximum(idr_asymptomatic, capped_delta_idr * minimum_asymptomatic_idr_fraction)
    idr_symptomatic = (capped_delta_idr - idr_asymptomatic * (1 - p_symptomatic_pre_omicron)) / p_symptomatic_pre_omicron
    idr_asymptomatic = np.minimum(idr_asymptomatic, maximum_asymptomatic_idr)
    omicron_idr = p_symptomatic_post_omicron * idr_symptomatic + (1 - p_symptomatic_post_omicron) * idr_asymptomatic
    sampled_ode_params['kappa_omicron_case'] = (omicron_idr / delta_idr).rename('kappa_omicron_case')

    ihr_scaling_factors = [
        # 1.4
        (  528, 1.4),  # Colorado
        (  529, 1.4),  # Connecticut
        (  531, 1.4),  # District of Columbia
        (  532, 1.4),  # Florida
        (  536, 1.4),  # Illinois
        (  543, 1.4),  # Maryland
        (  553, 1.4),  # New Jersey
        (  555, 1.4),  # New York
        (  558, 1.4),  # Ohio
        # 1.6
        (60358, 1.6),  # Aragon
        (60374, 1.6),  # Basque Country
        (60364, 1.6),  # Canary Islands
        (60359, 1.6),  # Cantabria
        (60360, 1.6),  # Castilla-La Mancha
        (60361, 1.6),  # Community of Madrid
        (60363, 1.6),  # Balearic Islands
        (60367, 1.6),  # Castile and León
        (60366, 1.6),  # Murcia
        # 2.0
        (43860, 2.0),  # Manitoba
        # 3.0
        (60370, 3.0),  # Navarre
        (60373, 3.0),  # Melilla
        (  196, 3.0),  # South Africa
    ]
    kappa_omicron_admission = pd.Series(
        sampled_ode_params['kappa_omicron_admission'],
        index=omicron_idr.index,
        name='kappa_omicron_admission'
    )
    for location_id, ihr_scaling_factor in ihr_scaling_factors:
        kappa_omicron_admission.loc[location_id] *= ihr_scaling_factor
    sampled_ode_params['kappa_omicron_admission'] = kappa_omicron_admission

    ifr_scaling_factors = [
        # 1.6
        (60358, 1.6),  # Aragon
        (60374, 1.6),  # Basque Country
        (60364, 1.6),  # Canary Islands
        (60359, 1.6),  # Cantabria
        (60360, 1.6),  # Castilla-La Mancha
        (60361, 1.6),  # Community of Madrid
        (60363, 1.6),  # Balearic Islands
        (60367, 1.6),  # Castile and León
        (60366, 1.6),  # Murcia
        # 3.0
        (60370, 3.0),  # Navarre
        (60373, 3.0),  # Melilla
        (60376, 3.0),  # La Rioja
        ( 4849, 3.0),  # Delhi
        ( 4863, 3.0),  # Mizoram
        (  196, 3.0),  # South Africa
        (  203, 3.0),  # Cabo Verde
        # 5.0
        (   50, 5.0),  # Montenegro
        (  176, 5.0),  # Comoros
        (  186, 5.0),  # Seychelles
        (  215, 5.0),  # Sao Tome and Principe
    ]
    kappa_omicron_death = pd.Series(
        sampled_ode_params['kappa_omicron_death'],
        index=omicron_idr.index,
        name='kappa_omicron_death'
    )
    for location_id, ifr_scaling_factor in ifr_scaling_factors:
        kappa_omicron_death.loc[location_id] *= ifr_scaling_factor
    sampled_ode_params['kappa_omicron_death'] = kappa_omicron_death

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
        daily_infections=(agg_first_pass_posterior_epi_measures['daily_naive_unvaccinated_infections']
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
        rates=second_pass_rates,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_waning_matrix,
        sampled_ode_params=sampled_ode_params,
        measure_downweights=specification.measure_downweights.to_dict(),
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

    out_params = ode_parameters.to_dict()['base_parameters']
    for name, duration in durations._asdict().items():
        out_params.loc[:, name] = duration

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

    second_pass_posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations
    )
    first_pass_posterior_epi_measures.loc[:, 'round'] = 1
    second_pass_posterior_epi_measures.loc[:, 'round'] = 2
    posterior_epi_measures = pd.concat([first_pass_posterior_epi_measures, second_pass_posterior_epi_measures])

    idx_cols = ['data_id', 'location_id', 'date', 'is_outlier']
    out_seroprevalence = seroprevalence.loc[:, idx_cols + ['reported_seroprevalence', 'seroprevalence']]
    # adjusted_seroprevalence['seroprevalence'] /= adjusted_seroprevalence['pct_unvaccinated']
    adjusted_seroprevalence = (adjusted_seroprevalence
                               .loc[:, idx_cols + ['seroprevalence']]
                               .rename(columns={'seroprevalence': 'adjusted_seroprevalence'}))
    out_seroprevalence = out_seroprevalence.merge(adjusted_seroprevalence)
    out_seroprevalence['sero_date'] = out_seroprevalence['date']
    out_seroprevalence['date'] -= pd.Timedelta(days=durations.exposure_to_seroconversion)

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

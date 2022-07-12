import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_past_infections(fit_version: str, draw_id: int, progress_bar: bool) -> None:
    logger.info(f'Starting past infections model for draw {draw_id}.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_cores = specification.workflow.task_specifications[FIT_JOBS.beta_fit].num_cores

    logger.info('Loading past infections data', context='read')
    mr_hierarchy = data_interface.load_hierarchy(name='mr')
    pred_hierarchy = data_interface.load_hierarchy(name='pred')
    risk_group_population = data_interface.load_population(measure='risk_group')
    epi_measures = data_interface.load_reported_epi_data()
    mortality_scalar = data_interface.load_total_covid_scalars(draw_id)['scalar']
    rhos = data_interface.load_variant_prevalence(scenario='reference')
    vaccinations = data_interface.load_vaccine_uptake(scenario='reference')
    etas = data_interface.load_vaccine_risk_reduction(scenario='reference')
    natural_waning_dist = data_interface.load_waning_parameters(measure='natural_waning_distribution').set_index('days')

    rates = []
    for measure in ['case', 'death', 'admission']:
        rates.append(get_round_data(data_interface.load_rates(
            measure_version=measure,
            draw_id=draw_id,
        )))
    rates = pd.concat(rates, axis=1)
    for level in ['parent_id', 'region_id', 'super_region_id', 'global']:
        rates = model.fill_from_hierarchy(rates, pred_hierarchy, level)

    logger.info('Sampling ODE parameters', context='transform')
    durations = model.sample_durations(specification.rates_parameters, draw_id, pred_hierarchy)
    variant_severity = model.sample_variant_severity(specification.rates_parameters, draw_id)
    _, natural_waning_matrix = model.sample_ode_params(
        variant_severity, specification.fit_parameters, draw_id
    )
    
    logger.info('Rescaling deaths and formatting epi measures', context='transform')
    epi_measures = model.filter_and_format_epi_measures(
        epi_measures=epi_measures,
        mortality_scalar=mortality_scalar,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        max_lag=durations.max_lag,
    )

    logger.info('Loading and resampling betas and infections.', context='transform')
    betas, infections, resampled_params, unrecoverable = model.load_and_resample_beta_and_infections(
        draw_id=draw_id,
        # This is sloppy, but we need to pull in data across a bunch of draws,
        # so this seems easiest.
        data_interface=data_interface,
    )

    logger.info('Computing composite beta', context='composite_spline')
    beta_fit_final, spline_infections, spline_infectious = model.build_composite_betas(
        betas=betas.drop(unrecoverable, axis=0),
        infections=infections.filter(like='total').rename(columns=lambda x: f'infection_{x.split("_")[-1]}'),
        alpha=resampled_params['alpha_all_infection'].groupby('location_id').mean(),
        num_cores=num_cores,
        progress_bar=progress_bar,    
    )
    
    logger.info('Prepping ODE fit parameters for past infections model.', context='transform')
    ode_parameters = model.prepare_past_infections_parameters(
        beta=beta_fit_final,
        rates=rates,
        durations=durations.to_ints(),
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_waning_matrix,
        resampled_params=resampled_params,
        hierarchy=pred_hierarchy,
        draw_id=draw_id,
    )
    
    logger.info('Building initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        measure='final',
        parameters=ode_parameters,
        rates=rates,
        population=risk_group_population,
        infections=spline_infections,    
    )
    
    logger.info('Running ODE fit', context='compute_ode')
    compartments, chis = model.run_posterior_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Prepping outputs.', context='transform')
    posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations.to_ints()
    )
    
    betas = pd.concat([betas, beta_fit_final], axis=1)
    out_params = ode_parameters.to_dict()['base_parameters']
    for name, duration in durations.to_dict().items():
        out_params.loc[:, name] = duration

    logger.info('Writing outputs', context='write')
    data_interface.save_fit_beta(betas, measure_version='final', draw_id=draw_id)
    data_interface.save_ode_params(out_params, measure_version='final', draw_id=draw_id)
    data_interface.save_input_epi_measures(epi_measures, measure_version='final', draw_id=draw_id)
    data_interface.save_phis(ode_parameters.phis, draw_id=draw_id)
    data_interface.save_rates(rates, measure_version='final', draw_id=draw_id)
    data_interface.save_compartments(compartments, measure_version='final', draw_id=draw_id)
    data_interface.save_posterior_epi_measures(infections, measure_version='resampled', draw_id=draw_id)
    data_interface.save_posterior_epi_measures(posterior_epi_measures, measure_version='final', draw_id=draw_id)

    logger.report()


def get_round_data(df: pd.DataFrame, round_id: int = 2) -> pd.DataFrame:
    return df[df['round'] == round_id].drop(columns='round')


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def past_infections(fit_version: str, draw_id: int,
                    progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_past_infections, logger, with_debugger)
    run(fit_version=fit_version,
        draw_id=draw_id,
        progress_bar=progress_bar)

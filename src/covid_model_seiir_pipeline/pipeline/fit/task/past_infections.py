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


def run_past_infections(fit_version: str, draw_id: int, progress_bar: bool) -> None:
    logger.info(f'Starting past infections model for draw {draw_id}.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_threads = specification.workflow.task_specifications[FIT_JOBS.beta_fit].num_cores

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
    betas = data_interface.load_fit_beta(draw_id=draw_id)['beta_all_infection']

    rates = []
    measure_kappas = []
    for measure in ['case', 'death', 'admission']:
        measure_rates = data_interface.load_rates(measure_version=measure, draw_id=draw_id)
        measure_rates = measure_rates.loc[measure_rates['round'] == 2].drop(columns='round')
        rates.append(measure_rates.sort_index())
        measure_kappa = data_interface.load_ode_params(measure_version=measure, draw_id=draw_id).filter(like='kappa').filter(like=measure)
        measure_kappas.append(measure_kappa)
    rates = pd.concat(rates, axis=1)
    measure_kappas = pd.concat(measure_kappas, axis=1)

    logger.info('Sampling ODE parameters', context='transform')
    durations = model.sample_durations(specification.rates_parameters, draw_id)
    variant_severity = model.sample_variant_severity(specification.rates_parameters, draw_id)
    sampled_ode_params, natural_waning_matrix = model.sample_ode_params(
        variant_severity, specification.fit_parameters, draw_id
    )

    logger.info('Rescaling deaths and formatting epi measures', context='transform')
    epi_measures = model.format_epi_measures(epi_measures, mr_hierarchy, pred_hierarchy, mortality_scalar, durations)

    logger.info('Prepping ODE fit parameters for second pass model.', context='transform')
    ode_parameters = model.prepare_past_infections_parameters(
        betas=betas,
        rates=rates,
        measure_kappas=measure_kappas,
        durations=durations,
        epi_measures=epi_measures,
        rhos=rhos,
        vaccinations=vaccinations,
        etas=etas,
        natural_waning_dist=natural_waning_dist,
        natural_waning_matrix=natural_waning_matrix,
        sampled_ode_params=sampled_ode_params,
        hierarchy=pred_hierarchy,
        draw_id=draw_id,
    )

    logger.info('Rebuilding initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        measure='final',
        parameters=ode_parameters,
        rates=rates,
        population=risk_group_population,
    )

    logger.info('Running second pass ODE fit', context='compute_ode')
    compartments, chis = model.run_posterior_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=num_threads,
        progress_bar=progress_bar,
    )
    posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations
    )

    logger.info('Prepping outputs.', context='transform')

    out_params = ode_parameters.to_dict()['base_parameters']
    for name, duration in durations._asdict().items():
        out_params.loc[:, name] = duration

    betas = betas.reindex(out_params.index)
    betas['beta'] = out_params['beta_all_infection']

    logger.info('Writing outputs', context='write')
    data_interface.save_ode_params(out_params, measure_version='final', draw_id=draw_id)
    data_interface.save_input_epi_measures(epi_measures, measure_version='final', draw_id=draw_id)
    data_interface.save_phis(ode_parameters.phis, draw_id=draw_id)
    data_interface.save_rates(rates, measure_version='final', draw_id=draw_id)
    data_interface.save_compartments(compartments, measure_version='final', draw_id=draw_id)
    data_interface.save_posterior_epi_measures(posterior_epi_measures, measure_version='final', draw_id=draw_id)
    data_interface.save_fit_beta(betas, measure_version='final', draw_id=draw_id)

    logger.report()


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

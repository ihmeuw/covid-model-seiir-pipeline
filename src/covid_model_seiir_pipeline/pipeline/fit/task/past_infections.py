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

    rates, kappas = [], []
    for measure in ['case', 'death', 'admission']:
        rates.append(get_round_data(data_interface.load_rates(
            measure_version=measure,
            draw_id=draw_id,
        )))
        kappas.append(data_interface.load_ode_params(
            measure_version=measure,
            draw_id=draw_id,
        ).filter(like='kappa').filter(like=measure))
    rates, kappas = [pd.concat(dfs, axis=1) for dfs in [rates, kappas]]
    for level in ['parent_id', 'region_id', 'super_region_id', 'global']:
        rates = model.fill_from_hierarchy(rates, pred_hierarchy, level)
        kappas = model.fill_from_hierarchy(kappas, pred_hierarchy, level)

    logger.info('Sampling ODE parameters', context='transform')
    durations = model.sample_durations(specification.rates_parameters, draw_id)
    variant_severity = model.sample_variant_severity(specification.rates_parameters, draw_id)
    sampled_ode_params, natural_waning_matrix = model.sample_ode_params(
        variant_severity, specification.fit_parameters, draw_id
    )

    logger.info('Loading and resampling betas and infections.', context='transform')
    betas, infections = model.load_and_resample_beta_and_infections(
        draw_id=draw_id,
        # This is sloppy, but we need to pull in data across a bunch of draws,
        # so this seems easiest.
        data_interface=data_interface,
    )

    logger.info('Computing composite beta', context='composite_spline')
    beta_fit_final = model.build_composite_betas(
        betas=betas,
        infections=infections.filter(like='total').rename(columns=lambda x: f'infection_{x.split("_")[-1]}'),
        alpha=sampled_ode_params['alpha_all_infection'],
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Rescaling deaths and formatting epi measures', context='transform')
    epi_measures = model.format_epi_measures(
        epi_measures=epi_measures,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        mortality_scalars=mortality_scalar,
        durations=durations,
    )

    logger.info('Prepping ODE fit parameters for second pass model.', context='transform')
    ode_parameters = model.prepare_past_infections_parameters(
        beta=beta_fit_final,
        rates=rates,
        measure_kappas=kappas,
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
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Prepping outputs.', context='transform')
    posterior_epi_measures = model.compute_posterior_epi_measures(
        compartments=compartments,
        durations=durations
    )
    betas = pd.concat([betas, beta_fit_final], axis=1)
    out_params = ode_parameters.to_dict()['base_parameters']
    for name, duration in durations._asdict().items():
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

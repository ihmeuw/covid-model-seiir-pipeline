import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    EPI_MEASURE_NAMES,
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

    logger.info('Loading rates data', context='read')
    mr_hierarchy = data_interface.load_hierarchy(name='mr')
    pred_hierarchy = data_interface.load_hierarchy(name='pred')
    five_year_population = data_interface.load_population(measure='five_year')
    total_population = data_interface.load_population(measure='total')
    epi_measures = data_interface.load_reported_epi_data()
    age_patterns = data_interface.load_age_patterns()
    seroprevalence = data_interface.load_seroprevalence(draw_id=draw_id).reset_index()
    sensitivity = data_interface.load_sensitivity(draw_id=draw_id)
    testing_capacity = data_interface.load_testing_data()
    covariate_pool = data_interface.load_covariate_options(draw_id=draw_id)
    rhos = data_interface.load_variant_prevalence(scenario='reference')
    variant_prevalence = rhos.sum(axis=1)
    mr_covariates = [data_interface.load_covariate(covariate, 'reference') for covariate in model.COVARIATE_POOL]

    import pdb; pdb.set_trace()
    logger.info('Sampling rates parameters', context='transform')
    durations = model.sample_durations(specification.rates_parameters, draw_id)
    variant_severity = model.sample_variant_severity(specification.rates_parameters, draw_id)
    day_inflection = model.sample_day_inflection(specification.rates_parameters, draw_id)
    logger.info('Subsetting seroprevalence for first pass rates model', context='transform')
    first_pass_seroprevalence = model.subset_seroprevalence(
        seroprevalence=seroprevalence,
        epi_data=epi_measures,
        variant_prevalence=variant_prevalence,
        population=total_population.population,
        params=specification.rates_parameters,
    )

    # dumb version of naive infections
    daily_deaths = epi_measures['daily_deaths'].dropna()
    naive_ifr = specification.rates_parameters.naive_ifr
    daily_infections = (daily_deaths / naive_ifr).rename('daily_infections').reset_index()
    daily_infections['date'] -= pd.Timedelta(days=durations.exposure_to_death)
    daily_infections = daily_infections.set_index(['location_id', 'date']).loc[:, 'daily_infections']

    logger.info('Running first-pass rates model', context='rates_model_1')
    rates = model.run_rates_pipeline(
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
    import pdb; pdb.set_trace()
    base_rates, epi_measures, smoothed_epi_measures, lags = model.run_rates_model(hierarchy)

    logger.info('Loading ODE fit input data', context='read')

    risk_group_pops = data_interface.load_population(measure='risk_group')
    rhos = data_interface.load_variant_prevalence(scenario='reference')
    vaccinations = data_interface.load_vaccine_uptake(scenario='reference')
    etas = data_interface.load_vaccine_risk_reduction(scenario='reference')
    natural_waning_dist = data_interface.load_waning_parameters(measure='natural_waning_distribution').set_index('days')
    natural_immunity_matrix = pd.DataFrame(
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
        base_rates,
        epi_measures,
        rhos,
        vaccinations,
        etas,
        natural_waning_dist,
        natural_immunity_matrix,
        regression_params,
        draw_id,
    )

    logger.info('Building initial condition.', context='transform')
    initial_condition = model.make_initial_condition(
        ode_parameters,
        base_rates,
        risk_group_pops,
    )

    logger.info('Running ODE fit', context='compute_ode')
    compartments, chis = model.run_ode_fit(
        initial_condition=initial_condition,
        ode_parameters=ode_parameters,
        num_cores=specification.workflow.task_specifications['beta_fit'].num_cores,
        progress_bar=progress_bar,
    )

    # Format and save data.
    logger.info('Prepping outputs', context='transform')
    epi_measures = pd.DataFrame(index=compartments.index)
    for measure in EPI_MEASURE_NAMES:
        cols = [f'{measure}_ancestral_all_{risk_group}' for risk_group in RISK_GROUP_NAMES]
        lag = lags.get(f'{measure}s', 0)
        epi_measures.loc[:, f'{measure}_naive_unvaccinated'] = (
            compartments
            .loc[:, cols]
            .sum(axis=1)
            .groupby('location_id')
            .apply(lambda x: x.reset_index(level='location_id', drop=True)
                              .shift(periods=lag, freq='D'))
        )
    epi_measures.loc[:, 'infection_naive'] = compartments.filter(like='NewENaive').sum(axis=1)
    epi_measures.loc[:, 'infection_total'] = compartments.filter(like='NewE').sum(axis=1)

    logger.info('Writing outputs', context='write')
    data_interface.save_epi_measures(epi_measures, draw_id=draw_id)

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

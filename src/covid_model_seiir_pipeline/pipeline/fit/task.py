import itertools

import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    REPORTED_EPI_MEASURE_NAMES
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
    epi_measures = pd.DataFrame(columns=REPORTED_EPI_MEASURE_NAMES, index=compartments.index)
    for measure in REPORTED_EPI_MEASURE_NAMES:
        cols = [f'{measure}_ancestral_all_{risk_group}' for risk_group in RISK_GROUP_NAMES]
        lag = lags[f'{measure}s']
        epi_measures.loc[:, measure] = (compartments
                                        .loc[:, cols]
                                        .sum(axis=1)
                                        .groupby('location_id')
                                        .apply(lambda x: x.reset_index(level='location_id', drop=True)
                                                          .shift(periods=lag, freq='D')))

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

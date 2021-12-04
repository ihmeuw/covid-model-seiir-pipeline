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


def run_beta_fit_postprocess(fit_version: str, measure: str, progress_bar: bool) -> None:
    logger.info(f'Starting beta fit postprocessing for measure {measure}.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_threads = specification.workflow.task_specifications[FIT_JOBS.beta_fit_postprocess].num_cores

    logger.info('Loading beta fit data', context='read')


    logger.info('Writing outputs', context='write')
    data_interface.save_ode_params(out_params, draw_id=draw_id)
    data_interface.save_input_epi_measures(epi_measures, draw_id=draw_id)
    data_interface.save_rates(prior_rates, draw_id=draw_id)
    data_interface.save_rates_data(rates_data, draw_id=draw_id)
    data_interface.save_posterior_epi_measures(posterior_epi_measures, draw_id=draw_id)
    data_interface.save_compartments(compartments, draw_id=draw_id)
    data_interface.save_beta(betas, draw_id=draw_id)
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


if __name__ == '__main__':
    beta_fit()

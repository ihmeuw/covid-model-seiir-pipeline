import functools
import multiprocessing

import click
import tqdm

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    parallel,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.specification import (
    PREPROCESSING_JOBS,
    PreprocessingSpecification,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model import serology

logger = cli_tools.task_performance_logger


def run_preprocess_serology(preprocessing_version: str, progress_bar: bool) -> None:
    logger.info(f'Starting preprocessing for serology_data.', context='setup')
    specification = PreprocessingSpecification.from_version_root(preprocessing_version)
    data_interface = PreprocessingDataInterface.from_specification(specification)
    num_cores = specification.workflow.task_specifications[PREPROCESSING_JOBS.preprocess_serology].num_cores

    logger.info('Loading raw serology input data', context='read')
    raw_seroprevalence = data_interface.load_raw_serology_data()
    hierarchy = data_interface.load_hierarchy('mr').reset_index()
    vaccine_data = data_interface.load_serology_vaccine_coverage()
    population = data_interface.load_population(measure='five_year')
    assay_map = data_interface.load_assay_map()

    logger.info('Cleaning serology input data', context='transform')
    seroprevalence = serology.process_raw_serology_data(raw_seroprevalence, hierarchy)
    seroprevalence = serology.assign_assay(seroprevalence, assay_map)
    vaccinated = serology.get_pop_vaccinated(population, vaccine_data)
    vaccinated['vaccinated'] *= specification.seroprevalence_parameters.vax_sero_prob

    logger.info('Sampling seroprevalence', context='model')
    seroprevalence_samples = serology.sample_seroprevalence(
        seroprevalence,
        n_samples=data_interface.get_n_total_draws(),
        bootstrap_samples=specification.seroprevalence_parameters.bootstrap_samples,
        correlate_samples=specification.seroprevalence_parameters.correlate_samples,
        num_threads=num_cores,
        progress_bar=progress_bar,
    )

    logger.info('Removing vaccinated from seroprevalence.', context='model')
    seroprevalence = serology.remove_vaccinated(seroprevalence, vaccinated)
    _runner = functools.partial(
        serology.remove_vaccinated,
        vaccinated=vaccinated,
    )
    seroprevalence_samples = parallel.run_parallel(
        _runner,
        arg_list=seroprevalence_samples,
        num_cores=num_cores,
        progress_bar=progress_bar
    )

    logger.info('Writing seroprevalence data.', context='write')
    data_interface.save_seroprevalence(seroprevalence)
    for draw, sample in tqdm.tqdm(enumerate(seroprevalence_samples),
                                  total=len(seroprevalence_samples), disable=not progress_bar):
        data_interface.save_seroprevalence(sample, draw_id=draw)

    logger.report()


@click.command()
@cli_tools.with_task_preprocessing_version
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def preprocess_serology(preprocessing_version: str,
                        progress_bar: bool,
                        verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_preprocess_serology, logger, with_debugger)
    run(preprocessing_version=preprocessing_version, progress_bar=progress_bar)

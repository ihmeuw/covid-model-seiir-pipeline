import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    aggregate,
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS
from covid_model_seiir_pipeline.pipeline.fit.model import postprocess

logger = cli_tools.task_performance_logger


def run_beta_fit_postprocess(fit_version: str, measure: str, progress_bar: bool) -> None:
    logger.info(f'Starting beta fit postprocessing for measure {measure}.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)

    measure_config = postprocess.MEASURES[measure]

    logger.info(f'Loading indexing data', context='read')
    input_epi_data = data_interface.load_input_epi_measures(draw_id=0)

    logger.info(f'Constructing index', context='transform')
    locs = input_epi_data.reset_index().location_id.unique().tolist()
    dates = input_epi_data.reset_index().date
    dates = pd.date_range(dates.min() - pd.Timedelta(days=30), dates.max())

    if measure_config.round_specific:
        idx = pd.MultiIndex.from_product([[1, 2], locs, dates], names=['round', 'location_id', 'date'])
    else:
        idx = pd.MultiIndex.from_product([locs, dates], names=['location_id', 'date'])

    logger.info(f'Loading {measure_config.label}.', context='read')
    num_cores = specification.workflow.task_specifications[FIT_JOBS.beta_fit_postprocess].num_cores
    measure_data = measure_config.loader(
        data_interface,
        index=idx,
        num_draws=data_interface.get_n_draws(),
        num_cores=num_cores,
        progress_bar=progress_bar,
    )

    if measure_config.aggregator is not None:
        logger.info(f'Aggregating to hierarchy', context='aggregate')
        hierarchy = data_interface.load_hierarchy('pred')
        population = data_interface.load_population('five_year')
        measure_data = measure_config.aggregator(measure_data, hierarchy, population)

    if measure_config.summary_metric:
        logger.info(f'Summarizing results for {measure}.', context='summarize')
        summarized = aggregate.summarize(measure_data,
                                         metric=measure_config.summary_metric,
                                         ci=measure_config.ci,
                                         ci2=measure_config.ci2)

        logger.info(f'Saving summaries for {measure}.', context='write_summary')
        data_interface.save_summary(summarized.reset_index(), measure_config.label)

    if measure_config.cumulative_label:
        cumulative_measure_data = measure_data.groupby(level='location_id').cumsum()
        logger.info(f'Summarizing results for cumulative {measure}.', context='summarize')
        summarized = aggregate.summarize(cumulative_measure_data,
                                         metric=measure_config.summary_metric,
                                         ci=measure_config.ci,
                                         ci2=measure_config.ci2)

        logger.info(f'Saving summaries for cumulative {measure}.', context='write_summary')
        data_interface.save_summary(summarized.reset_index(), measure_config.cumulative_label)

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_measure
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_fit_postprocess(fit_version: str, measure: str,
                         progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_fit_postprocess, logger, with_debugger)
    run(fit_version=fit_version,
        measure=measure,
        progress_bar=progress_bar)

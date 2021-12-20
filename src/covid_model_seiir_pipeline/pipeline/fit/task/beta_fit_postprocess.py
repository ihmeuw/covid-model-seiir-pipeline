from typing import Tuple, Union

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
    logger.info(f'Starting postprocessing for version {fit_version}.', context='setup')

    config = postprocess.MEASURES[measure]
    if isinstance(config, postprocess.MeasureConfig):
        postprocess_measure(fit_version, config, progress_bar)
    elif isinstance(config, postprocess.CompositeMeasureConfig):
        postprocess_composite_measure(fit_version, config, progress_bar)
    else:
        raise NotImplementedError(f'Unknown measure {measure}.')

    logger.report()


def postprocess_measure(fit_version: str, measure_config: postprocess.MeasureConfig, progress_bar: bool) -> None:
    specification, data_interface = build_spec_and_data_interface(fit_version)

    measure_data = load_measure(
        measure_config,
        specification,
        data_interface,
        progress_bar,
    )
    measure_data = do_aggregation(
        measure_data,
        measure_config,
        data_interface,
    )
    summarize_and_write(
        measure_data,
        measure_config,
        data_interface,
    )


def postprocess_composite_measure(fit_version: str,
                                  measure_config: postprocess.CompositeMeasureConfig,
                                  progress_bar: bool) -> None:
    specification, data_interface = build_spec_and_data_interface(fit_version)

    measure_data = {}
    for base_measure, base_measure_config in measure_config.base_measures.items():
        base_measure_data = load_measure(
            base_measure_config,
            specification,
            data_interface,
            progress_bar,
        )
        base_measure_data = do_aggregation(
            base_measure_data,
            base_measure_config,
            data_interface,
        )
        measure_data[base_measure] = base_measure_data

    if measure_config.duration_label:
        measure_data['duration'] = load_measure(
            postprocess.MEASURES['ode_parameters'],
            specification,
            data_interface,
            progress_bar,
        )

    measure_data = measure_config.combiner(**measure_data)

    summarize_and_write(
        measure_data,
        measure_config,
        data_interface,
    )


def build_spec_and_data_interface(fit_version: str) -> Tuple[FitSpecification, FitDataInterface]:
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    return specification, data_interface


def load_measure(measure_config: postprocess.MeasureConfig,
                 specification: FitSpecification,
                 data_interface: FitDataInterface,
                 progress_bar: bool) -> pd.DataFrame:
    logger.info(f'Loading indexing data', context='read')
    input_epi_data = data_interface.load_input_epi_measures(draw_id=0)

    logger.info(f'Constructing index', context='transform')
    locs = input_epi_data.reset_index().location_id.unique().tolist()
    dates = input_epi_data.reset_index().date
    dates = pd.date_range(dates.min() - pd.Timedelta(days=30), dates.max())

    if measure_config.round_specific:
        idx = pd.MultiIndex.from_product([locs, dates, [1, 2]], names=['location_id', 'date', 'round'])
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
    return measure_data


def do_aggregation(measure_data: pd.DataFrame,
                   measure_config: postprocess.MeasureConfig,
                   data_interface: FitDataInterface) -> pd.DataFrame:
    if measure_config.aggregator is not None:
        logger.info(f'Aggregating to hierarchy', context='aggregate')
        hierarchy = data_interface.load_hierarchy('pred')
        population = data_interface.load_population('five_year')
        measure_data = measure_config.aggregator(measure_data, hierarchy, population)
    return measure_data


def summarize_and_write(measure_data: pd.DataFrame,
                        measure_config: Union[postprocess.MeasureConfig, postprocess.CompositeMeasureConfig],
                        data_interface: FitDataInterface):
    to_write = {measure_config.label: measure_data}

    if measure_config.cumulative_label:
        logger.info(f'Generating cumulative results for {measure_config.label}.', context='cumsum')
        agg_levels = [c for c in list(measure_data.columns) + list(measure_data.index.names)
                      if c in ['location_id', 'round']]
        cumulative_measure_data = (measure_data
                                   .reset_index()
                                   .set_index(agg_levels + ['date'])
                                   .groupby(agg_levels)
                                   .cumsum())
        to_write[measure_config.cumulative_label] = cumulative_measure_data

    for label, data in to_write.items():
        if measure_config.summary_metric:
            logger.info(f'Summarizing results for {label}.', context='summarize')
            data = aggregate.summarize(data,
                                       metric=measure_config.summary_metric,
                                       ci=measure_config.ci,
                                       ci2=measure_config.ci2)

        logger.info(f'Saving data for {label}.', context='write')
        data_interface.save_summary(data, label)


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

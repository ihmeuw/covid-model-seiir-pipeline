from typing import List, Union
from pathlib import Path

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.specification import (
    SplicingSpecification,
    AggregationSpecification,
    PostprocessingSpecification,
    POSTPROCESSING_JOBS,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.postprocessing import model


logger = cli_tools.task_performance_logger


def do_splicing(measure_data: pd.DataFrame,
                measure_config: Union[model.MeasureConfig, model.CovariateConfig],
                splicing_configs: List[SplicingSpecification],
                data_interface: PostprocessingDataInterface,
                scenario_name: str) -> pd.DataFrame:
    if measure_config.splice:
        for splicing_config in splicing_configs:
            logger.info(f'Splicing in results from {splicing_config.output_version}.', context='splice')
            try:
                previous_data = data_interface.load_previous_version_output_draws(splicing_config.output_version,
                                                                                  scenario_name,
                                                                                  measure_config.label)
                measure_data = model.splice_data(measure_data, previous_data, splicing_config.locations)
            except FileNotFoundError:
                logger.warning(f"Can't find {measure_config.label} data for {splicing_config.output_version}. "
                               f"Skipping splicing for this version.")
    return measure_data


def do_aggregation(measure_data: pd.DataFrame,
                   measure_config: Union[model.MeasureConfig, model.CovariateConfig, model.MiscellaneousConfig],
                   aggregation_configs: List[AggregationSpecification],
                   data_interface: PostprocessingDataInterface) -> pd.DataFrame:
    if measure_config.aggregator is not None and aggregation_configs:
        for aggregation_config in aggregation_configs:
            logger.info(f'Aggregating to hierarchy {aggregation_config.to_dict()}', context='aggregate')
            hierarchy = data_interface.load_aggregation_heirarchy(aggregation_config)
            population = data_interface.load_populations()
            measure_data = measure_config.aggregator(measure_data, hierarchy, population)
    return measure_data


def summarize_and_write(measure_data: pd.DataFrame,
                        measure_config: Union[model.MeasureConfig, model.CovariateConfig, model.MiscellaneousConfig],
                        data_interface: PostprocessingDataInterface,
                        measure: str, scenario_name: str):
    logger.info(f'Summarizing results for {measure}.', context='summarize')
    summarized = model.summarize(measure_data)
    if measure_config.write_draws:
        logger.info(f'Saving draws for {measure}.', context='write_draws')
        data_interface.save_output_draws(measure_data.reset_index(), scenario_name, measure)
    logger.info(f'Saving summaries for {measure}.', context='write_summary')
    data_interface.save_output_summaries(summarized.reset_index(), scenario_name, measure)


def postprocess_measure(postprocessing_spec: PostprocessingSpecification,
                        data_interface: PostprocessingDataInterface,
                        scenario_name: str, measure: str) -> None:
    measure_config = model.MEASURES[measure]
    logger.info(f'Loading {measure}.', context='read')
    num_cores = postprocessing_spec.workflow.task_specifications[POSTPROCESSING_JOBS.postprocess].num_cores
    measure_data = measure_config.loader(scenario_name, data_interface, num_cores)
    if isinstance(measure_data, (list, tuple)):
        logger.info(f'Concatenating {measure}.', context='concatenate')
        measure_data = pd.concat(measure_data, axis=1)

    logger.info(f'Resampling {measure}.', context='resample')
    measure_data = model.resample_draws(measure_data, data_interface.load_resampling_map())

    measure_data = do_splicing(
        measure_data,
        measure_config,
        postprocessing_spec.splicing,
        data_interface,
        scenario_name,
    )
    measure_data = do_aggregation(
        measure_data,
        measure_config,
        postprocessing_spec.aggregation,
        data_interface,
    )

    summarize_and_write(
        measure_data,
        measure_config,
        data_interface,
        measure_config.label,
        scenario_name,
    )

    if measure_config.calculate_cumulative:
        cumulative_measure_data = measure_data.groupby(level='location_id').cumsum()
        summarize_and_write(
            cumulative_measure_data,
            measure_config,
            data_interface,
            measure_config.cumulative_label,
            scenario_name,
        )


def postprocess_composite_measure(postprocessing_spec: PostprocessingSpecification,
                                  data_interface: PostprocessingDataInterface,
                                  scenario_name: str, measure: str) -> None:
    measure_config = model.COMPOSITE_MEASURES[measure]
    logger.info(f'Loading inputs for {measure}.', context='read')
    num_cores = postprocessing_spec.workflow.task_specifications[POSTPROCESSING_JOBS.postprocess].num_cores
    measure_data = {}
    for base_measure, base_measure_config in measure_config.base_measures.items():
        base_measure_data = base_measure_config.loader(scenario_name, data_interface, num_cores)

        if isinstance(base_measure_data, (list, tuple)):
            logger.info(f'Concatenating {base_measure}.', context='concatenate')
            base_measure_data = pd.concat(base_measure_data, axis=1)

        logger.info(f'Resampling {base_measure}.', context='resample')
        base_measure_data = model.resample_draws(base_measure_data, data_interface.load_resampling_map())

        base_measure_data = do_splicing(
            base_measure_data,
            base_measure_config,
            postprocessing_spec.splicing,
            data_interface,
            scenario_name,
        )
        base_measure_data = do_aggregation(
            base_measure_data,
            base_measure_config,
            postprocessing_spec.aggregation,
            data_interface,
        )
        measure_data[base_measure] = base_measure_data

    measure_data = measure_config.combiner(**measure_data)

    summarize_and_write(
        measure_data,
        measure_config,
        data_interface,
        measure_config.label,
        scenario_name
    )


def postprocess_covariate(postprocessing_spec: PostprocessingSpecification,
                          data_interface: PostprocessingDataInterface,
                          scenario_name: str, covariate: str) -> None:
    covariate_config = model.COVARIATES[covariate]
    logger.info(f'Loading {covariate} data.', context='read')
    num_cores = postprocessing_spec.workflow.task_specifications[POSTPROCESSING_JOBS.postprocess].num_cores
    covariate_data = covariate_config.loader(covariate, covariate_config.time_varying,
                                             scenario_name, data_interface, num_cores)

    covariate_version = data_interface.get_covariate_version(covariate, scenario_name)
    n_draws = data_interface.get_n_draws()

    input_covariate_data = data_interface.load_input_covariate(covariate, covariate_version)
    covariate_observed = input_covariate_data.reset_index(level='observed')
    covariate_observed['observed'] = covariate_observed['observed'].fillna(0)

    logger.info(f'Concatenating {covariate}.', context='concatenate')
    covariate_data = pd.concat(covariate_data, axis=1)
    logger.info('Resampling draws', context='resample')
    covariate_data = model.resample_draws(covariate_data,
                                          data_interface.load_resampling_map())

    covariate_data = do_splicing(
        covariate_data,
        covariate_config,
        postprocessing_spec.splicing,
        data_interface,
        scenario_name,
    )
    covariate_data = do_aggregation(
        covariate_data,
        covariate_config,
        postprocessing_spec.aggregation,
        data_interface,
    )

    logger.info(f'Combining {covariate} forecasts with history.', context='merge_history')
    covariate_data = covariate_data.merge(covariate_observed, left_index=True,
                                          right_index=True, how='outer').reset_index()
    draw_cols = [f'draw_{i}' for i in range(n_draws)]
    if 'date' in covariate_data.columns:
        index_cols = ['location_id', 'date', 'observed']
    else:
        index_cols = ['location_id', 'observed']

    covariate_data = covariate_data.set_index(index_cols)[draw_cols]
    covariate_data['modeled'] = covariate_data.notnull().all(axis=1).astype(int)

    input_covariate = pd.concat([covariate_observed.reset_index().set_index(index_cols)] * n_draws, axis=1)
    input_covariate.columns = draw_cols
    covariate_data = covariate_data.combine_first(input_covariate).set_index('modeled', append=True)

    summarize_and_write(
        covariate_data,
        covariate_config,
        data_interface,
        covariate_config.label,
        scenario_name
    )


def postprocess_miscellaneous(postprocessing_spec: PostprocessingSpecification,
                              data_interface: PostprocessingDataInterface,
                              scenario_name: str, measure: str):
    miscellaneous_config = model.MISCELLANEOUS[measure]
    logger.info(f'Loading {measure}.', context='read')
    location_ids = data_interface.load_location_ids()
    miscellaneous_data = miscellaneous_config.loader(data_interface)

    if miscellaneous_config.is_cumulative:
        logger.info('Filling missing dates.', context='fill_dates')
        miscellaneous_data = miscellaneous_data.loc[location_ids]
        miscellaneous_data = model.fill_cumulative_date_index(miscellaneous_data)

    miscellaneous_data = do_aggregation(
        miscellaneous_data,
        miscellaneous_config,
        postprocessing_spec.aggregation,
        data_interface,
    )

    if miscellaneous_config.is_table:
        miscellaneous_data = miscellaneous_data.reset_index()

    logger.info(f'Saving {measure} data.', context='write')
    data_interface.save_output_miscellaneous(miscellaneous_data,
                                             scenario_name,
                                             miscellaneous_config.label,
                                             miscellaneous_config.is_table)


def run_seir_postprocessing(postprocessing_version: str, scenario: str, measure: str) -> None:
    logger.info(f'Starting postprocessing for version {postprocessing_version}, scenario {scenario}.', context='setup')
    postprocessing_spec = PostprocessingSpecification.from_path(
        Path(postprocessing_version) / static_vars.POSTPROCESSING_SPECIFICATION_FILE
    )
    data_interface = PostprocessingDataInterface.from_specification(postprocessing_spec)

    if measure in model.MEASURES:
        postprocess_measure(postprocessing_spec, data_interface, scenario, measure)
    elif measure in model.COMPOSITE_MEASURES:
        postprocess_composite_measure(postprocessing_spec, data_interface, scenario, measure)
    elif measure in model.COVARIATES:
        postprocess_covariate(postprocessing_spec, data_interface, scenario, measure)
    elif measure in model.MISCELLANEOUS:
        postprocess_miscellaneous(postprocessing_spec, data_interface, scenario, measure)
    else:
        raise NotImplementedError(f'Unknown measure {measure}.')

    logger.report()


@click.command()
@cli_tools.with_task_postprocessing_version
@cli_tools.with_scenario
@cli_tools.with_measure
@cli_tools.add_verbose_and_with_debugger
def postprocess(postprocessing_version: str, scenario: str, measure: str,
                verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_seir_postprocessing, logger, with_debugger)
    run(postprocessing_version=postprocessing_version,
        scenario=scenario,
        measure=measure)


if __name__ == '__main__':
    postprocess()

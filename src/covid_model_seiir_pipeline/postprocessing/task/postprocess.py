from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Optional

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.postprocessing.specification import PostprocessingSpecification, POSTPROCESSING_JOBS
from covid_model_seiir_pipeline.postprocessing.model import final_outputs, resampling, aggregators, splicing


def postprocess_measure(postprocessing_spec: PostprocessingSpecification,
                        data_interface: PostprocessingDataInterface,
                        scenario_name: str, measure: str) -> None:
    measure_config = final_outputs.MEASURES[measure]
    logger.info(f'Loading {measure}.')
    num_cores = postprocessing_spec.workflow.task_specifications[POSTPROCESSING_JOBS.postprocess].num_cores
    measure_data = measure_config.loader(scenario_name, data_interface, num_cores)
    if isinstance(measure_data, (list, tuple)):
        logger.info(f'Concatenating {measure}.')
        measure_data = pd.concat(measure_data, axis=1)

    logger.info(f'Resampling {measure}.')
    measure_data = resampling.resample_draws(measure_data,
                                             data_interface.load_resampling_map())

    if measure_config.splice:
        for splicing_config in postprocessing_spec.splicing:
            previous_data = data_interface.load_previous_version_output_draws(splicing_config.output_version,
                                                                              scenario_name,
                                                                              measure_config.label)
            measure_data = splicing.splice_data(measure_data, previous_data, splicing_config.locations)

    if measure_config.aggregator is not None and postprocessing_spec.aggregation:
        for aggregation_config in postprocessing_spec.aggregation:
            hierarchy = data_interface.load_aggregation_heirarchy(aggregation_config)
            population = data_interface.load_populations()
            measure_data = measure_config.aggregator(measure_data, hierarchy, population)

    logger.info(f'Saving draws and summaries for {measure}.')
    data_interface.save_output_draws(measure_data.reset_index(), scenario_name, measure_config.label)
    summarized = aggregators.summarize(measure_data)
    data_interface.save_output_summaries(summarized.reset_index(), scenario_name, measure_config.label)

    if measure_config.calculate_cumulative:
        logger.info(f'Saving cumulative draws and summaries for {measure}.')
        cumulative_measure_data = measure_data.groupby(level='location_id').cumsum()
        data_interface.save_output_draws(cumulative_measure_data.reset_index(), scenario_name,
                                         measure_config.cumulative_label)
        summarized = aggregators.summarize(cumulative_measure_data)
        data_interface.save_output_summaries(summarized.reset_index(), scenario_name,
                                             measure_config.cumulative_label)


def postprocess_covariate(postprocessing_spec: PostprocessingSpecification,
                          data_interface: PostprocessingDataInterface,
                          scenario_name: str, covariate: str) -> None:
    covariate_config = final_outputs.COVARIATES[covariate]
    logger.info(f'Loading {covariate}.')
    num_cores = postprocessing_spec.workflow.task_specifications[POSTPROCESSING_JOBS.postprocess].num_cores
    covariate_data = covariate_config.loader(covariate, covariate_config.time_varying,
                                             scenario_name, data_interface, num_cores)

    logger.info(f'Concatenating and resampling {covariate}.')
    covariate_data = pd.concat(covariate_data, axis=1)
    covariate_data = resampling.resample_draws(covariate_data,
                                               data_interface.load_resampling_map())

    if covariate_config.splice:
        for locs_to_splice, splice_version in postprocessing_spec.splicing:
            previous_data = data_interface.load_previous_version_output_draws(splice_version,
                                                                              scenario_name,
                                                                              covariate_config.label)
            covariate_data = splicing.splice_data(covariate_data, previous_data, locs_to_splice)

    if covariate_config.aggregator is not None:
        for aggregation_config in postprocessing_spec.aggregation:
            hierarchy = data_interface.load_aggregation_heirarchy(aggregation_config)
            population = data_interface.load_populations()
            covariate_data = covariate_config.aggregator(covariate_data, hierarchy, population)

    covariate_version = data_interface.get_covariate_version(covariate, scenario_name)
    location_ids = data_interface.load_location_ids()
    n_draws = data_interface.get_n_draws()

    logger.info(f'Loading and processing input data for {covariate}.')
    input_covariate_data = data_interface.load_input_covariate(covariate, covariate_version, location_ids)
    covariate_observed = input_covariate_data.reset_index(level='observed')
    covariate_observed['observed'] = covariate_observed['observed'].fillna(0)

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

    logger.info(f'Saving data for {covariate}.')
    if covariate_config.draw_level:
        data_interface.save_output_draws(covariate_data.reset_index(), scenario_name, covariate_config.label)

    summarized_data = aggregators.summarize(covariate_data)
    data_interface.save_output_summaries(summarized_data.reset_index(), scenario_name, covariate_config.label)


def postprocess_miscellaneous(postprocessing_spec: PostprocessingSpecification,
                              data_interface: PostprocessingDataInterface,
                              scenario_name: str, measure: str):
    miscellaneous_config = final_outputs.MISCELLANEOUS[measure]
    logger.info(f'Loading {measure}.')
    miscellaneous_data = miscellaneous_config.loader(data_interface)

    if miscellaneous_config.aggregator is not None:
        for aggregation_config in postprocessing_spec.aggregation:
            hierarchy = data_interface.load_aggregation_heirarchy(aggregation_config)
            population = data_interface.load_populations()
            miscellaneous_data = miscellaneous_config.aggregator(miscellaneous_data, hierarchy, population)
    if miscellaneous_config.is_table:
        miscellaneous_data = miscellaneous_data.reset_index()

    logger.info(f'Saving {measure} data.')
    data_interface.save_output_miscellaneous(miscellaneous_data,
                                             scenario_name,
                                             miscellaneous_config.label,
                                             miscellaneous_config.is_table)


def run_seir_postprocessing(postprocessing_version: str, scenario_name: str, measure: str) -> None:
    logger.info(f'Starting postprocessing for version {postprocessing_version}, scenario {scenario_name}.')
    postprocessing_spec = PostprocessingSpecification.from_path(
        Path(postprocessing_version) / static_vars.POSTPROCESSING_SPECIFICATION_FILE
    )
    data_interface = PostprocessingDataInterface.from_specification(postprocessing_spec)

    if measure in final_outputs.MEASURES:
        postprocess_measure(postprocessing_spec, data_interface, scenario_name, measure)
    elif measure in final_outputs.COVARIATES:
        postprocess_covariate(postprocessing_spec, data_interface, scenario_name, measure)
    elif measure in final_outputs.MISCELLANEOUS:
        postprocess_miscellaneous(postprocessing_spec, data_interface, scenario_name, measure)
    else:
        raise NotImplementedError(f'Unknown measure {measure}.')

    logger.info('**DONE**')


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--postprocessing-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)
    parser.add_argument("--measure", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(verbose=1)  # Debug level
    args = parse_arguments()
    run_seir_postprocessing(postprocessing_version=args.postprocessing_version,
                            scenario_name=args.scenario_name,
                            measure=args.measure)


if __name__ == '__main__':
    main()

from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Any, Callable, Dict, List, Optional

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline import io
from covid_model_seiir_pipeline.io.keys import MetadataKey
from covid_model_seiir_pipeline.forecasting.specification import (
    ForecastSpecification,
    ScenarioSpecification,
    FORECAST_JOBS,
)
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting import postprocessing_lib as pp


def postprocess_measure(data_interface: ForecastDataInterface,
                        resampling_map: Dict[int, Dict[str, List[int]]],
                        scenario_name: str, measure: str,
                        num_cores: int) -> None:
    measure_config = MEASURES[measure]
    logger.info(f'Loading {measure}.')
    measure_data = measure_config.loader(scenario_name, data_interface, num_cores)
    if isinstance(measure_data, (list, tuple)):
        logger.info(f'Concatenating {measure}.')
        measure_data = pd.concat(measure_data, axis=1)
    logger.info(f'Resampling {measure}.')
    measure_data = pp.resample_draws(measure_data, resampling_map)

    if measure_config.aggregator is not None:
        hierarchy = pp.load_modeled_hierarchy(data_interface)
        population = pp.load_populations(data_interface)
        measure_data = measure_config.aggregator(measure_data, hierarchy, population)

    logger.info(f'Saving draws and summaries for {measure}.')
    data_interface.save_output_draws(measure_data.reset_index(), scenario_name, measure_config.label)
    summarized = pp.summarize(measure_data)
    data_interface.save_output_summaries(summarized.reset_index(), scenario_name, measure_config.label)

    if measure_config.calculate_cumulative:
        logger.info(f'Saving cumulative draws and summaries for {measure}.')
        cumulative_measure_data = measure_data.groupby(level='location_id').cumsum()
        data_interface.save_output_draws(cumulative_measure_data.reset_index(), scenario_name,
                                         measure_config.cumulative_label)
        summarized = pp.summarize(cumulative_measure_data)
        data_interface.save_output_summaries(summarized.reset_index(), scenario_name,
                                             measure_config.cumulative_label)


def postprocess_covariate(data_interface: ForecastDataInterface,
                          resampling_map: Dict[int, Dict[str, List[int]]],
                          scenario_spec: ScenarioSpecification,
                          scenario_name: str, covariate: str,
                          num_cores: int) -> None:
    covariate_config = COVARIATES[covariate]
    logger.info(f'Loading {covariate}.')
    covariate_data = covariate_config.loader(covariate, covariate_config.time_varying,
                                             scenario_name, data_interface, num_cores)
    logger.info(f'Concatenating and resampling {covariate}.')
    covariate_data = pd.concat(covariate_data, axis=1)
    covariate_data = pp.resample_draws(covariate_data, resampling_map)

    if covariate_config.aggregator is not None:
        hierarchy = pp.load_modeled_hierarchy(data_interface)
        population = pp.load_populations(data_interface)
        covariate_data = covariate_config.aggregator(covariate_data, hierarchy, population)

    covariate_version = scenario_spec.covariates[covariate]
    location_ids = data_interface.load_location_ids()
    n_draws = data_interface.get_n_draws()

    logger.info(f'Loading and processing input data for {covariate}.')
    input_covariate_data = data_interface.load_covariate(covariate, covariate_version, location_ids, with_observed=True)
    covariate_observed = input_covariate_data.reset_index(level='observed')
    covariate_data = covariate_data.merge(covariate_observed, left_index=True,
                                          right_index=True, how='outer').reset_index()
    draw_cols = [f'draw_{i}' for i in range(n_draws)]
    if 'date' in covariate_data.columns:
        index_cols = ['location_id', 'date', 'observed']
    else:
        index_cols = ['location_id', 'observed']

    covariate_data = covariate_data.set_index(index_cols)[draw_cols]
    covariate_data['modeled'] = covariate_data.notnull().all(axis=1).astype(int)

    input_covariate = pd.concat([input_covariate_data.reorder_levels(index_cols)] * n_draws, axis=1)
    input_covariate.columns = draw_cols
    covariate_data = covariate_data.combine_first(input_covariate).set_index('modeled', append=True)

    logger.info(f'Saving data for {covariate}.')
    if covariate_config.draw_level:
        data_interface.save_output_draws(covariate_data.reset_index(), scenario_name, covariate_config.label)

    summarized_data = pp.summarize(covariate_data)
    data_interface.save_output_summaries(summarized_data.reset_index(), scenario_name, covariate_config.label)


def postprocess_miscellaneous(data_interface: ForecastDataInterface,
                              scenario_name: str, measure: str,
                              num_cores: int):
    miscellaneous_config = MISCELLANEOUS[measure]
    logger.info(f'Loading {measure}.')
    miscellaneous_data = miscellaneous_config.loader(data_interface)

    if miscellaneous_config.aggregator is not None:
        hierarchy = pp.load_modeled_hierarchy(data_interface)
        population = pp.load_populations(data_interface)
        miscellaneous_data = miscellaneous_config.aggregator(miscellaneous_data, hierarchy, population)

    logger.info(f'Saving {measure} data.')
    if miscellaneous_config.is_table:
        data_interface.save_output_miscellaneous(miscellaneous_data.reset_index(), scenario_name,
                                                 miscellaneous_config.label)
    else:
        # FIXME: Still sad about this.
        key = MetadataKey(root=data_interface.forecast_root._root / scenario_name,
                          disk_format='yaml',
                          data_type=miscellaneous_config.label)
        io.dump(miscellaneous_data, key)


def run_seir_postprocessing(postprocessing_version: str, scenario_name: str, measure: str) -> None:
    logger.info(f'Starting postprocessing for forecast version {forecast_version}, scenario {scenario_name}.')
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    resampling_map = data_interface.load_resampling_map()
    num_cores = forecast_spec.workflow.task_specifications[FORECAST_JOBS.postprocess].num_cores

    if measure in MEASURES:
        postprocess_measure(data_interface, resampling_map, scenario_name, measure, num_cores)
    elif measure in COVARIATES:
        postprocess_covariate(data_interface, resampling_map, scenario_spec, scenario_name, measure, num_cores)
    elif measure in MISCELLANEOUS:
        postprocess_miscellaneous(data_interface, scenario_name, measure, num_cores)
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

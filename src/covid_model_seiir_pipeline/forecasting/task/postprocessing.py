from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Any, Callable, Dict, List, Optional

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import (
    ForecastSpecification,
    ScenarioSpecification,
    PostprocessingSpecification
)
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting import postprocessing_lib as pp


class MeasureConfig:
    def __init__(self,
                 loader: Callable[[str, ForecastDataInterface], Any],
                 label: str,
                 calculate_cumulative: bool = False,
                 cumulative_label: str = None,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.calculate_cumulative = calculate_cumulative
        self.cumulative_label = cumulative_label
        self.aggregator = aggregator


class CovariateConfig:
    def __init__(self,
                 loader: Callable[[str, bool, str, ForecastDataInterface], List[pd.Series]],
                 label: str,
                 time_varying: bool = False,
                 draw_level: bool = False,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.time_varying = time_varying
        self.draw_level = draw_level
        self.aggregator = aggregator


class OtherConfig:
    def __init__(self,
                 loader: Callable[[ForecastDataInterface], Any],
                 label: str,
                 is_table: bool = True,
                 aggregator: Callable = None):
        self.loader = loader
        self.label = label
        self.is_table = is_table
        self.aggregator = aggregator


MEASURES = {
    'deaths': MeasureConfig(
        pp.load_deaths,
        'daily_deaths',
        calculate_cumulative=True,
        cumulative_label='cumulative_deaths',
        aggregator=pp.sum_aggregator,
    ),
    'infections': MeasureConfig(
        pp.load_infections,
        'daily_infections',
        calculate_cumulative=True,
        cumulative_label='cumulative_infections',
        aggregator=pp.sum_aggregator,
    ),
    'r_effective': MeasureConfig(
        pp.load_r_effective,
        'r_effective',
    ),
    'betas': MeasureConfig(
        pp.load_betas,
        'betas',
    ),
    'beta_residuals': MeasureConfig(
        pp.load_beta_residuals,
        'log_beta_residuals',
    ),
    'coefficients': MeasureConfig(
        pp.load_coefficients,
        'coefficients',
    ),
    'scaling_parameters': MeasureConfig(
        pp.load_scaling_parameters,
        'beta_scaling_parameters'
    ),
    'elastispliner_noisy': MeasureConfig(
        pp.load_es_noisy,
        'daily_elastispliner_noisy',
        calculate_cumulative=True,
        cumulative_label='cumulative_elastispliner_noisy',
        aggregator=pp.sum_aggregator,
    ),
    'elastispliner_smoothed': MeasureConfig(
        pp.load_es_smoothed,
        'daily_elastispliner_smoothed',
        calculate_cumulative=True,
        cumulative_label='cumulative_elastispliner_smoothed',
        aggregator=pp.sum_aggregator,
    )
}


COVARIATES = {
    'mobility': CovariateConfig(
        pp.load_covariate,
        'mobility',
        time_varying=True,
        draw_level=True,
        aggregator=pp.mean_aggregator,
    ),
    'testing': CovariateConfig(
        pp.load_covariate,
        'testing',
        time_varying=True,
        aggregator=pp.mean_aggregator,
    ),
    'pneumonia': CovariateConfig(
        pp.load_covariate,
        'pneumonia',
        time_varying=True,
        # aggregator=pp.mean_aggregator,  This probably isn't what we want.  Still in discussion.
    ),
    'mask_use': CovariateConfig(
        pp.load_covariate,
        'mask_use',
        time_varying=True,
        aggregator=pp.mean_aggregator,
    ),
    'air_pollution_pm_2_5': CovariateConfig(
        pp.load_covariate,
        'air_pollution_pm_2_5',
        aggregator=pp.mean_aggregator,
    ),
    'lri_mortality': CovariateConfig(
        pp.load_covariate,
        'lri_mortality',
        aggregator=pp.mean_aggregator,
    ),
    'proportion_over_2_5k': CovariateConfig(
        pp.load_covariate,
        'proportion_over_2_5k',
        aggregator=pp.mean_aggregator,
    ),
    'proportion_under_100m': CovariateConfig(
        pp.load_covariate,
        'proportion_under_100m',
        aggregator=pp.mean_aggregator,
    ),
    'smoking_prevalence': CovariateConfig(
        pp.load_covariate,
        'smoking_prevalence',
        aggregator=pp.mean_aggregator,
    ),
}

MISCELLANEOUS = {
    'full_data': OtherConfig(
        pp.load_full_data,
        'full_data',
        aggregator=pp.sum_aggregator,
    ),
    'elastispliner_inputs': OtherConfig(
        pp.load_elastispliner_inputs,
        'full_data_es_processed',
        aggregator=pp.sum_aggregator,
    ),
    'version_map': OtherConfig(
        pp.build_version_map,
        'version_map',
    ),
    'populations': OtherConfig(
        pp.load_populations,
        'populations',
    ),
    'hierarchy': OtherConfig(
        pp.load_hierarchy,
        'hierarchy',
    ),
    'locations_modeled_and_missing': OtherConfig(
        pp.get_locations_modeled_and_missing,
        'modeled_and_missing_locations',
        is_table=False,
    ),
}


def postprocess_measure(data_interface: ForecastDataInterface,
                        resampling_map: Dict[int, Dict[str, List[int]]],
                        scenario_name: str, measure: str) -> None:
    measure_config = MEASURES[measure]
    logger.info(f'Loading {measure}.')
    measure_data = measure_config.loader(scenario_name, data_interface)
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
                          scenario_name: str, covariate: str) -> None:
    covariate_config = COVARIATES[covariate]
    logger.info(f'Loading {covariate}.')
    covariate_data = covariate_config.loader(covariate, covariate_config.time_varying, scenario_name, data_interface)
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
                              scenario_name: str, measure: str):
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
        # FIXME: yuck
        miscellaneous_dir = data_interface.forecast_paths.scenario_paths[scenario_name].output_miscellaneous
        measure_path = miscellaneous_dir / f'{miscellaneous_config.label}.yaml'
        with measure_path.open('w') as f:
            yaml.dump(miscellaneous_data, f)


def run_seir_postprocessing(output_version: str, scenario_name: str, measure: str) -> None:
    logger.info(f'Starting postprocessing for output version {output_version}, scenario {scenario_name}.')
    postprocessing_spec = PostprocessingSpecification.from_path(
        Path(output_version) / 'postprocessing_specification.yaml'
    )
    forecast_spec = ForecastSpecification.from_path(
        Path(postprocessing_spec.data.forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec, postprocessing_spec)
    resampling_map = data_interface.load_resampling_map()

    if measure in MEASURES:
        postprocess_measure(data_interface, resampling_map, scenario_name, measure)
    elif measure in COVARIATES:
        postprocess_covariate(data_interface, resampling_map, scenario_spec, scenario_name, measure)
    elif measure in MISCELLANEOUS:
        postprocess_miscellaneous(data_interface, scenario_name, measure)
    else:
        raise NotImplementedError(f'Unknown measure {measure}.')

    logger.info('**DONE**')


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--output-version", type=str, required=True)
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
    run_seir_postprocessing(output_version=args.output_version,
                            scenario_name=args.scenario_name,
                            measure=args.measure)


if __name__ == '__main__':
    main()

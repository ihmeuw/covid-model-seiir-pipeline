from argparse import ArgumentParser, Namespace
from pathlib import Path
import shlex
from typing import Optional

from covid_shared.cli_tools.logging import configure_logging_to_terminal
from loguru import logger
import pandas as pd
import yaml

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting import postprocessing_lib as pp


def run_seir_postprocessing(forecast_version: str, scenario_name: str) -> None:
    logger.info(f'Starting postprocessing for forecast version {forecast_version}, scenario {scenario_name}.')
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    resampling_map = data_interface.load_resampling_map()
    logger.info('Loading SEIR outputs')
    deaths, infections, r_effective = pp.load_output_data(scenario_name, data_interface)
    betas = pp.load_betas(scenario_name, data_interface)
    beta_residuals = pp.load_beta_residuals(data_interface)
    coefficients = pp.load_coefficients(data_interface)
    scaling_parameters = pp.load_scaling_parameters(scenario_name, data_interface)

    logger.info('Concatenating and resampling SEIR outputs.')
    measures = [deaths, infections, r_effective, betas, beta_residuals, coefficients, scaling_parameters]
    measures = [pd.concat(m, axis=1) for m in measures]
    measures = pp.resample_draws(resampling_map, *measures)
    deaths, infections, r_effective, betas, beta_residuals, coefficients, scaling_parameters = measures
    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    cumulative_infections = infections.groupby(level='location_id').cumsum()

    logger.info('Loading SEIR covariates')
    all_covs = scenario_spec.covariates
    time_varying_covs = ['mobility', 'mask_use', 'testing', 'pneumonia']
    non_time_varying_covs = set(all_covs).difference(time_varying_covs + ['intercept'])
    cov_order = {'time_varying': time_varying_covs, 'non_time_varying': non_time_varying_covs}
    covariates = pp.load_covariates(scenario_name, cov_order, data_interface)

    logger.info('Concatenating and resampling SEIR covariates')
    location_ids = data_interface.load_location_ids()
    n_draws = data_interface.get_n_draws()
    for cov_name, covariate in covariates.items():
        logger.info(f'Concatenating and resampling {cov_name}.')
        covariate = pd.concat(covariate, axis=1)
        covariate = pp.resample_draws(resampling_map, covariate)[0]
        input_covariate = data_interface.load_covariate(cov_name, all_covs[cov_name],
                                                        location_ids, with_observed=True)
        covariate_observed = input_covariate.reset_index(level='observed')
        covariate = covariate.merge(covariate_observed, left_index=True, right_index=True, how='outer').reset_index()

        draw_cols = [f'draw_{i}' for i in range(n_draws)]
        index_cols = ['location_id', 'date', 'observed'] if 'date' in covariate.columns else ['location_id', 'observed']
        covariate = covariate.set_index(index_cols)[draw_cols]
        covariate['modeled'] = covariate.notnull().all(axis=1).astype(int)
        input_covariate = pd.concat([input_covariate.reorder_levels(index_cols)] * n_draws, axis=1)
        input_covariate.columns = draw_cols
        covariate = covariate.combine_first(input_covariate).set_index('modeled', append=True)
        covariates[cov_name] = covariate

    logger.info('Loading other data sources.')
    version_map = pp.build_version_map(data_interface)
    es_inputs = pp.load_elastispliner_inputs(data_interface)
    full_data = pp.load_full_data(data_interface)
    populations = pp.load_populations(data_interface)
    hierarchy, locations_modeled_and_missing = pp.load_location_information(data_interface)
    es_noisy, es_smoothed = pp.load_elastispliner_outputs(data_interface)

    logger.info('Resampling other data sources')
    es_noisy = es_noisy.rename(columns={f'draw_{i}': i for i in range(n_draws)})
    es_smoothed = es_smoothed.rename(columns={f'draw_{i}': i for i in range(n_draws)})
    es_noisy, es_smoothed = pp.resample_draws(resampling_map, es_noisy, es_smoothed)
    es_noisy_daily = es_noisy.groupby(level='location_id').apply(lambda x: x - x.shift(fill_value=0))
    es_smoothed_daily = es_smoothed.groupby(level='location_id').apply(lambda x: x - x.shift(fill_value=0))

    output_draws = {
        'daily_deaths': deaths,
        'daily_infections': infections,
        'r_effective': r_effective,
        'cumulative_deaths': cumulative_deaths,
        'cumulative_infections': cumulative_infections,
        'betas': betas,
        'log_beta_residuals': beta_residuals,
        'coefficients': coefficients,
        'beta_scaling_parameters': scaling_parameters,
        'cumulative_elastispliner_noisy': es_noisy,
        'cumulative_elastispliner_smoothed': es_smoothed,
        'daily_elastispliner_noisy': es_noisy_daily,
        'daily_elastispliner_smoothed': es_smoothed_daily,
        'mobility': covariates['mobility'],
    }
    logger.info('Saving SEIR output draws and summaries.')
    for measure, data in output_draws.items():
        logger.info(f'Saving {measure} data.')
        data_interface.save_output_draws(data.reset_index(), scenario_name, measure)
        summarized_data = pp.summarize(data)
        data_interface.save_output_summaries(summarized_data.reset_index(), scenario_name, measure)

    del covariates['mobility']
    output_no_draws = {
        **covariates,
    }
    logger.info('Saving SEIR covariate summaries.')
    for measure, data in output_no_draws.items():
        logger.info(f'Saving {measure} data.')
        summarized_data = pp.summarize(data)
        data_interface.save_output_summaries(summarized_data.reset_index(), scenario_name, measure)

    miscellaneous_outputs = {
        'full_data_raw': full_data,
        'full_data_es_processed': es_inputs,
        'populations': populations,
        'version_map': version_map,
        'hierarchy': hierarchy
    }
    logger.info('Saving miscellaneous outputs.')
    for measure, data in miscellaneous_outputs.items():
        logger.info(f'Saving {measure} data.')
        data_interface.save_output_miscellaneous(data.reset_index(), scenario_name, measure)

    # FIXME: yuck
    miscellaneous_dir = data_interface.forecast_paths.scenario_paths[scenario_name].output_miscellaneous
    modeled_and_missing_path = miscellaneous_dir / 'modeled_and_missing_locations.yaml'
    with modeled_and_missing_path.open('w') as f:
        yaml.dump(locations_modeled_and_missing, f)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    logger.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    configure_logging_to_terminal(1)
    args = parse_arguments()
    run_seir_postprocessing(forecast_version=args.forecast_version,
                            scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()

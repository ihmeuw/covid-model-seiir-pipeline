from argparse import ArgumentParser, Namespace
import functools
import logging
import multiprocessing
from pathlib import Path
import shlex
from typing import Optional, List

import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting import postprocessing_lib as pp
from covid_model_seiir_pipeline.forecasting.workflow import FORECAST_SCALING_CORES

log = logging.getLogger(__name__)


def run_seir_postprocessing(forecast_version: str, scenario_name: str) -> None:
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    resampling_map = data_interface.load_resampling_map()
    deaths, infections, r_effective = pp.load_output_data(scenario_name, data_interface)
    betas = pp.load_betas(scenario_name, data_interface)
    beta_residuals = pp.load_beta_residuals(data_interface)

    all_covs = scenario_spec.covariates
    time_varying_covs = ['mobility', 'mask_use', 'testing', 'pneumonia']
    non_time_varying_covs = set(all_covs).difference(time_varying_covs + ['intercept'])
    cov_order = {'time_varying': time_varying_covs, 'non_time_varying': non_time_varying_covs}
    covariates = pp.load_covariates(scenario_name, cov_order, data_interface)

    measures = [deaths, infections, r_effective, betas, beta_residuals]
    measures = [pd.concat(m, axis=1) for m in measures]
    measures = resample_draws(resampling_map, *measures)
    deaths, infections, r_effective, betas, beta_residuals = measures

    location_ids = data_interface.load_location_ids()
    n_draws = data_interface.get_n_draws()
    for cov_name, covariate in covariates.items():
        covariate = pd.concat(covariate, axis=1).reset_index()
        input_covariate = data_interface.load_covariate(cov_name, all_covs[cov_name],
                                                        location_ids, with_observed=True)
        input_covariate = input_covariate.reset_index(level='observed')
        covariate['observed'] = input_covariate['observed']
        covariate.set_index(covariate.columns.difference(list(range(n_draws))))
        covariate = resample_draws(resampling_map, covariate)
        covariates[cov_name] = covariate

    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    cumulative_infections = infections.groupby(level='location_id').cumsum()
    output_draws = {
        'daily_deaths': deaths,
        'daily_infections': infections,
        'r_effective': r_effective,
        'cumulative_deaths': cumulative_deaths,
        'cumulative_infections': cumulative_infections,
        'betas': betas,
        'log_beta_residuals': beta_residuals,
        'mobility': covariates['mobility'],
    }
    for measure, data in output_draws.items():
        data_interface.save_output_draws(data.reset_index(), scenario_name, measure)
        summarized_data = summarize(data)
        data_interface.save_output_summaries(summarized_data.reset_index(), scenario_name, measure)

    del covariates['mobility']
    output_no_draws = {
        **covariates,
    }
    for measure, data in output_no_draws.items():
        summarized_data = summarize(data)
        data_interface.save_output_summaries(summarized_data.reset_index(), scenario_name, measure)


def summarize(data: pd.DataFrame):
    data['mean'] = data.mean(axis=1)
    data['upper'] = data.quantile(.975, axis=1)
    data['lower'] = data.quantile(.025, axis=1)
    return data[['mean', 'upper', 'lower']]


def concat_measures(*measure_data: List[pd.Series]):
    _runner = functools.partial(
        pd.concat,
        axis=1
    )
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        measures = pool.map(_runner, measure_data)
    return measures


def resample_draws(resampling_map, *measure_data: pd.DataFrame):
    _runner = functools.partial(
        resample_draws_by_measure,
        resampling_map=resampling_map
    )
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        resampled = pool.map(_runner, measure_data)
    return resampled


def resample_draws_by_measure(measure_data: pd.DataFrame, resampling_map):
    output = []
    for location_id, loc_map in resampling_map.items():
        loc_data = measure_data.loc[location_id]
        loc_data[loc_map['to_resample']] = loc_data[loc_map['to_fill']]
        loc_data = pd.concat({location_id: loc_data}, names=['location_id'])
        output.append(loc_data)

    resampled = pd.concat(output)
    resampled.columns = [f'draw_{draw}' for draw in resampled.columns]
    return resampled


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
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
    args = parse_arguments()
    run_seir_postprocessing(forecast_version=args.forecast_version,
                            scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()

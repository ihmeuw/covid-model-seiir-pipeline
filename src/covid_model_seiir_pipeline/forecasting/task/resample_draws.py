from argparse import ArgumentParser, Namespace
import functools
import logging
import multiprocessing
from pathlib import Path
import shlex
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import FORECAST_SCALING_CORES

log = logging.getLogger(__name__)


def run_resample(forecast_version: str) -> None:
    forecast_spec = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    resampling_params = forecast_spec.postprocessing.resampling
    resampling_map = build_resampling_map(resampling_params, data_interface)
    data_interface.save_resampling_map(resampling_map)


def build_resampling_map(resampling_params: Dict, data_interface: ForecastDataInterface):
    resampling_ref_scenario = resampling_params['reference_scenario']
    deaths, *_ = load_output_data(resampling_ref_scenario, data_interface)
    deaths = concat_measures(deaths)[0]
    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    max_deaths = cumulative_deaths.groupby(level='location_id').max()
    upper_deaths = max_deaths.quantile(resampling_params['upper_quantile'], axis=1)
    lower_deaths = max_deaths.quantile(resampling_params['lower_quantile'], axis=1)
    resample_map = {}
    for location_id in max_deaths.index:
        upper, lower = upper_deaths.at[location_id], lower_deaths.at[location_id]
        loc_deaths = max_deaths.loc[location_id]
        to_resample = loc_deaths[(upper < loc_deaths) | (loc_deaths < lower)].index.tolist()
        to_fill = np.random.choice(loc_deaths.index, len(to_resample), replace=False)
        resample_map[location_id] = (to_resample, to_fill)
    return resample_map


def load_output_data(scenario: str, data_interface: ForecastDataInterface):
    _runner = functools.partial(
        load_output_data_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = list(pool.map(_runner, draws))
    deaths, infections, r_effective = zip(*outputs)

    return deaths, infections, r_effective


def load_output_data_by_draw(draw_id: int, scenario: str, data_interface: ForecastDataInterface):
    draw_df = data_interface.load_raw_outputs(scenario, draw_id)
    draw_df = draw_df.set_index(['location_id', 'date']).sort_index()
    deaths = draw_df['deaths'].rename(draw_id)
    infections = draw_df['infections'].rename(draw_id)
    r_effective = draw_df['r_effective'].rename(draw_id)
    return deaths, infections, r_effective


def concat_measures(*measure_data: List[pd.Series]) -> List[pd.DataFrame]:
    _runner = functools.partial(
        pd.concat,
        axis=1
    )
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        measure_data = pool.map(_runner, measure_data)
    return measure_data


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    run_resample(forecast_version=args.forecast_version)


if __name__ == '__main__':
    main()

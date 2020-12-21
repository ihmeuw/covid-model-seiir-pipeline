import functools
import multiprocessing
from typing import List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface


def load_deaths(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    deaths, *_ = load_output_data(scenario, data_interface, num_cores)
    return deaths


def load_infections(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _, infections, *_ = load_output_data(scenario, data_interface, num_cores)
    return infections


def load_r_effective(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _, _, r_effective = load_output_data(scenario, data_interface, num_cores)
    return r_effective


def load_output_data(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_raw_outputs,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    deaths, infections, r_effective = zip(*outputs)

    return deaths, infections, r_effective


def load_coefficients(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(data_interface.load_regression_coefficients, draws)
    return outputs


def load_scaling_parameters(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_scaling_parameters,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_covariate(covariate: str, time_varying: bool, scenario: str,
                   data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    _runner = functools.partial(
        data_interface.load_covariate,
        covariate=covariate,
        time_varying=time_varying,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)

    return outputs


def load_betas(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    _runner = functools.partial(
        data_interface.load_betas,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        betas = pool.map(_runner, draws)
    return betas


def load_beta_residuals(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        beta_residuals = pool.map(data_interface.load_beta_residuals, draws)
    return beta_residuals


def load_elastispliner_inputs(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    return data_interface.load_elastispliner_inputs()


def load_es_noisy(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_elastispliner_outputs(data_interface, noisy=True)


def load_es_smoothed(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_elastispliner_outputs(data_interface, noisy=False)


def load_elastispliner_outputs(data_interface: 'PostprocessingDataInterface', noisy: bool):
    return data_interface.load_elastispliner_outputs(noisy)


def load_full_data(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    return data_interface.load_full_data()


def build_version_map(data_interface: 'PostprocessingDataInterface') -> pd.Series:
    return data_interface.build_version_map()


def load_populations(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_populations()


def load_hierarchy(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_hierarchy()


def get_locations_modeled_and_missing(data_interface: 'PostprocessingDataInterface'):
    return data_interface.get_locations_modeled_and_missing()

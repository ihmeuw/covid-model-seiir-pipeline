import functools
import multiprocessing
from typing import List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface


def load_deaths(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'deaths', data_interface, num_cores)


def load_infections(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'infections', data_interface, num_cores)


def load_r_controlled(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'r_controlled', data_interface, num_cores)


def load_r_effective(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'r_effective', data_interface, num_cores)


def load_herd_immunity(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'herd_immunity', data_interface, num_cores)


def load_total_susceptible(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'total_susceptible', data_interface, num_cores)


def load_total_immune(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'total_immune', data_interface, num_cores)


def load_hospital_admissions(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'hospital_admissions', data_interface, num_cores)


def load_icu_admissions(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'icu_admissions', data_interface, num_cores)


def load_hospital_census(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'hospital_census', data_interface, num_cores)


def load_icu_census(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'icu_census', data_interface, num_cores)


def load_ventilator_census(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return load_output_data(scenario, 'ventilator_census', data_interface, num_cores)


def load_hospital_correction_factors(data_interface: 'PostprocessingDataInterface'):
    dfs = []
    for correction_type in ['hospital', 'icu', 'ventilator']:
        df = data_interface.load_raw_outputs(
            draw_id=0,  # All draws are identical.
            scenario='worse',  # All scenarios the same.
            measure=f'{correction_type}_census_correction',
        )
        df = df.rename(f'{correction_type}_census')
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def load_raw_census_data(data_interface: 'PostprocessingDataInterface'):
    census_data = data_interface.load_hospital_census_data()
    return pd.concat([
        data.rename(census_type) for census_type, data in census_data.to_dict().items()
    ], axis=1)


def load_output_data(scenario: str, measure: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_raw_outputs,
        scenario=scenario,
        measure=measure,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


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


def load_full_data(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    return data_interface.load_full_data()


def load_age_specific_deaths(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    full_data = data_interface.load_full_data()
    total_deaths = (full_data
                    .groupby('location_id')
                    .cumulative_deaths
                    .max()
                    .dropna())
    mortality_ratio = data_interface.load_mortality_ratio()
    age_specific_deaths = (mortality_ratio * total_deaths).rename('age_specific_deaths').to_frame()
    return age_specific_deaths


def build_version_map(data_interface: 'PostprocessingDataInterface') -> pd.Series:
    return data_interface.build_version_map()


def load_populations(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_populations()


def load_hierarchy(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_hierarchy()


def get_locations_modeled_and_missing(data_interface: 'PostprocessingDataInterface'):
    return data_interface.get_locations_modeled_and_missing()

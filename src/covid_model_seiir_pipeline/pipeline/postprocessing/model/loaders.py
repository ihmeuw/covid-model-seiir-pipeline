import functools
import multiprocessing
from typing import List, TYPE_CHECKING

import pandas as pd

from covid_model_seiir_pipeline.lib.aggregate import summarize

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface


def load_deaths(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_raw_output_deaths,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_unscaled_deaths(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    death_draws = load_deaths(scenario, data_interface, num_cores)
    mortality_scalars = data_interface.load_total_covid_scalars()
    mortality_scalars = mortality_scalars.loc[death_draws[0].index]
    unscaled_deaths = [deaths / mortality_scalars.loc[:, f'draw_{draw}']
                       for draw, deaths in enumerate(death_draws)]
    return unscaled_deaths


def load_output_data(output_name: str, fallback: str = None):
    def inner(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
        try:
            return _load_output_data(scenario, output_name, data_interface, num_cores)
        except ValueError:
            if fallback is None:
                raise
            return _load_output_data(scenario, fallback, data_interface, num_cores)

    return inner


def load_vaccine_summaries(output_name: str):
    def inner(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
        n_draws = data_interface.get_n_draws()
        summary = data_interface.load_vaccination_summaries(output_name)
        summary = pd.concat([summary]*n_draws, axis=1)    
        summary.columns = list(range(n_draws))
        return summary
    return inner


def load_vaccine_efficacy_table(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_waning_parameters('base_vaccine_efficacy')


def load_variant_prevalence(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_variant_prevalence('reference')


def load_ode_params(output_name: str):
    def inner(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
        return _load_ode_params(scenario, output_name, data_interface, num_cores)
    return inner


def load_beta_residuals(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    _runner = functools.partial(
        data_interface.load_beta_residuals,
        scenario=scenario,
    )

    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        beta_residuals = pool.map(_runner, draws)
    return beta_residuals


def load_scaled_beta_residuals(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    _runner = functools.partial(
        data_interface.load_scaled_beta_residuals,
        scenario=scenario,
    )

    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        beta_residuals = pool.map(_runner, draws)
    return beta_residuals


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


def load_coefficients(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(data_interface.load_coefficients, draws)
    return outputs


def load_excess_mortality_scalars(data_interface: 'PostprocessingDataInterface'):
    return summarize(data_interface.load_total_covid_scalars())


def load_raw_census_data(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_hospital_census_data()


def load_hospital_correction_factors(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_hospital_correction_factors()


def load_hospital_bed_capacity(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_hospital_bed_capacity()


def load_scaling_parameters(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_beta_scales,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_full_data_unscaled(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    full_data = data_interface.load_reported_epi_data().reset_index()
    location_ids = data_interface.load_location_ids()
    full_data = full_data[full_data.location_id.isin(location_ids)].set_index(['location_id', 'date'])
    return full_data


def load_total_covid_deaths(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> pd.DataFrame:
    full_data = load_full_data_unscaled(data_interface)
    deaths = full_data['cumulative_deaths'].dropna()
    deaths = deaths.groupby('location_id').diff().fillna(deaths)
    mortality_scalars = data_interface.load_total_covid_scalars().loc[deaths.index]
    scaled_deaths = mortality_scalars.mul(deaths, axis=0).groupby('location_id').cumsum().fillna(0.0)
    scaled_deaths.columns = [int(c.split('_')[1]) for c in scaled_deaths]
    return scaled_deaths


def build_version_map(data_interface: 'PostprocessingDataInterface') -> pd.Series:
    return data_interface.build_version_map()


def load_populations(data_interface: 'PostprocessingDataInterface'):
    idx_cols = ['location_id', 'sex_id', 'year_id', 
                'age_group_id', 'age_group_years_start', 'age_group_years_end']
    return data_interface.load_population('all_population').reset_index().set_index(idx_cols)


def load_hierarchy(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_hierarchy('pred')


def get_locations_modeled_and_missing(data_interface: 'PostprocessingDataInterface'):
    return data_interface.get_locations_modeled_and_missing()


#########################
# Non-interface methods #
#########################

def _load_output_data(scenario: str, measure: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_single_raw_output,
        scenario=scenario,
        measure=measure,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def _load_ode_params(scenario: str, measure: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_single_ode_param,
        scenario=scenario,
        measure=measure,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs

import functools
import multiprocessing
from typing import List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface

##########
# Deaths #
##########


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
    em_scalars = load_excess_mortality_scalars(data_interface)
    em_scalars = em_scalars.reindex(death_draws[0].index).groupby('location_id').fillna(method='ffill')
    unscaled_deaths = [d / em_scalars for d in death_draws]
    return unscaled_deaths


def load_deaths_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'modeled_deaths_wild', data_interface, num_cores)


def load_deaths_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'modeled_deaths_variant', data_interface, num_cores)


##############
# Infections #
##############

def load_infections(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'infections', data_interface, num_cores)


def load_infections_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'modeled_infections_wild', data_interface, num_cores)


def load_infections_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'modeled_infections_variant', data_interface, num_cores)


def load_infections_natural_breakthrough(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'natural_immunity_breakthrough', data_interface, num_cores)


def load_infections_vaccine_breakthrough(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccine_breakthrough', data_interface, num_cores)


#########
# Cases #
#########

def load_cases(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'cases', data_interface, num_cores)


####################
# Hospitalizations #
####################

def load_hospital_admissions(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'hospital_admissions', data_interface, num_cores)


def load_icu_admissions(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'icu_admissions', data_interface, num_cores)


def load_hospital_census(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'hospital_census', data_interface, num_cores)


def load_icu_census(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'icu_census', data_interface, num_cores)


def load_ventilator_census(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'ventilator_census', data_interface, num_cores)


################
# Vaccinations #
################

def load_effectively_vaccinated(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_effectively_vaccinated,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_vaccinations_immune_all(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccinations_immune_all', data_interface, num_cores)


def load_vaccinations_immune_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccinations_immune_wild', data_interface, num_cores)


def load_vaccinations_protected_all(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccinations_protected_all', data_interface, num_cores)


def load_vaccinations_protected_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccinations_protected_wild', data_interface, num_cores)


def load_vaccinations_effective(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccinations_effective', data_interface, num_cores)


def load_vaccinations_ineffective(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'vaccinations_ineffective', data_interface, num_cores)


######################
# Susceptible/Immune #
######################

def load_total_susceptible_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'total_susceptible_wild', data_interface, num_cores)


def load_total_susceptible_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'total_susceptible_variant', data_interface, num_cores)


def load_total_immune_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'total_immune_wild', data_interface, num_cores)


def load_total_immune_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'total_immune_variant', data_interface, num_cores)


#####################
# Other Epi metrics #
#####################

def load_r_controlled_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'r_controlled_wild', data_interface, num_cores)


def load_r_effective_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'r_effective_wild', data_interface, num_cores)


def load_r_controlled_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'r_controlled_variant', data_interface, num_cores)


def load_r_effective_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'r_effective_variant', data_interface, num_cores)


def load_r_effective(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'r_effective', data_interface, num_cores)


def load_beta(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    return _load_ode_params(scenario, 'beta', data_interface, num_cores)


def load_beta_hat(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    return _load_ode_params(scenario, 'beta_hat', data_interface, num_cores)


def load_beta_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    return _load_ode_params(scenario, 'beta_wild', data_interface, num_cores)


def load_beta_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    return _load_ode_params(scenario, 'beta_variant', data_interface, num_cores)


def load_empirical_beta(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'beta', data_interface, num_cores)


def load_empirical_beta_wild(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'beta_wild', data_interface, num_cores)


def load_empirical_beta_variant(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'beta_variant', data_interface, num_cores)


def load_beta_residuals(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int) -> List[pd.Series]:
    _runner = functools.partial(
        data_interface.load_beta_residuals,
        scenario=scenario,
    )

    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        beta_residuals = pool.map(_runner, draws)
    return beta_residuals


def load_non_escape_variant_prevalence(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_ode_params(scenario, 'rho', data_interface, num_cores)


def load_escape_variant_prevalence(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_ode_params(scenario, 'rho_variant', data_interface, num_cores)


def load_empirical_escape_variant_prevalence(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    return _load_output_data(scenario, 'variant_prevalence', data_interface, num_cores)


##############
# Covariates #
##############

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
        outputs = pool.map(data_interface.load_regression_coefficients, draws)
    return outputs


#########################
# Miscellaneous metrics #
#########################

def load_excess_mortality_scalars(data_interface: 'PostprocessingDataInterface'):
    return data_interface.load_excess_mortality_scalars()


def load_raw_census_data(data_interface: 'PostprocessingDataInterface'):
    census_data = data_interface.load_hospital_census_data()
    return pd.concat([
        data.rename(census_type) for census_type, data in census_data.to_dict().items()
    ], axis=1)


def load_scaling_parameters(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    _runner = functools.partial(
        data_interface.load_scaling_parameters,
        scenario=scenario,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_full_data(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    return data_interface.load_full_data()


def load_unscaled_full_data(data_interface: 'PostprocessingDataInterface') -> pd.DataFrame:
    full_data = data_interface.load_full_data()
    em_scalars = load_excess_mortality_scalars(data_interface)
    import pdb; pdb.set_trace()


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


def load_ifr_es(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(data_interface.load_ifr, draws)
    return outputs


def load_ihr_es(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(data_interface.load_ihr, draws)
    return outputs


def load_idr_es(scenario: str, data_interface: 'PostprocessingDataInterface', num_cores: int):
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(num_cores) as pool:
        outputs = pool.map(data_interface.load_idr, draws)
    return outputs


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

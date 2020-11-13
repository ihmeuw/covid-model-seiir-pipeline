from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Type
import itertools

import numpy as np
from odeopt.ode import RK4, ODESolver
import pandas as pd

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.math import compute_beta_hat
from covid_model_seiir_pipeline.forecasting.data import ScenarioData
from covid_model_seiir_pipeline.forecasting.specification import ScenarioSpecification


def forecast_beta(covariates: pd.DataFrame,
                  coefficients: pd.DataFrame,
                  beta_shift_parameters: pd.DataFrame) -> pd.DataFrame:
    log_beta_hat = compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    # Rescale the predictions of beta based on the residuals from the
    # regression.
    betas = _beta_shift(beta_hat, beta_shift_parameters).set_index('location_id')
    return betas


def get_population_partition(population: pd.DataFrame,
                             location_ids: List[int],
                             population_partition: str) -> Dict[str, pd.Series]:
    """Create a location-specific partition of the population.

    Parameters
    ----------
    population
        A dataframe with location, age, and sex specific populations.
    location_ids
        A list of location ids used in the regression model.
    population_partition
        A string describing how the population should be partitioned.

    Returns
    -------
        A mapping between the SEIR compartment suffix for the partition groups
        and a series mapping location ids to the proportion of people in each
        compartment that should be allocated to the partition group.

    """
    if population_partition == 'none':
        partition_map = {
            '': pd.Series(1.0, index=location_ids)
        }
    elif population_partition == 'old_and_young':
        modeled_locations = population['location_id'].isin(location_ids)
        is_2019 = population['year_id']
        is_both_sexes = population['sex_id'] == 3
        five_year_bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
        is_five_year_bins = population['age_group_id'].isin(five_year_bins)
        population = population.loc[modeled_locations & is_2019 & is_both_sexes & is_five_year_bins, :]

        total_pop = population.groupby('location_id')['population'].sum()
        young_pop = population[population['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
        old_pop = total_pop - young_pop

        partition_map = {
            'y': young_pop / total_pop,
            'o': old_pop / total_pop,
        }
    else:
        raise NotImplementedError

    return partition_map


class CompartmentInfo:

    def __init__(self,
                 compartments: List[str],
                 group_suffixes: List[str]):
        self.compartments = compartments
        self.group_suffixes = group_suffixes


def get_past_components(beta_regression_df: pd.DataFrame,
                        population_partition: Dict[str, pd.Series],
                        ode_system: str) -> Tuple[CompartmentInfo, pd.DataFrame]:
    regression_compartments = static_vars.SEIIR_COMPARTMENTS
    system_compartment_map = {
        'normal': _split_compartments(static_vars.SEIIR_COMPARTMENTS, population_partition),
        'vaccine': _split_compartments(static_vars.VACCINE_SEIIR_COMPARTMENTS, population_partition),
    }
    system_compartments = system_compartment_map[ode_system]

    beta_regression_df = (beta_regression_df
                          .set_index(['location_id', 'date'])
                          .sort_index())
    past_beta = beta_regression_df['beta']
    past_components = beta_regression_df[regression_compartments]

    partitioned_past_components = []
    for compartment in regression_compartments:
        for partition_group, proportion in population_partition.items():
            partitioned_past_components.append(
                (past_components[compartment] * proportion).rename(f'{compartment}_{partition_group}')
            )
    past_components = pd.concat(partitioned_past_components, axis=1)

    rows_to_fill = past_components.notnull().all(axis=1)
    compartments_to_fill = system_compartments.difference(past_components.columns)
    past_components = past_components.reindex(system_compartments, axis=1)
    for compartment_to_fill in compartments_to_fill:
        past_components.loc[rows_to_fill, compartment_to_fill] = 0

    past_components = pd.concat([past_beta, past_components], axis=1).reset_index(level='date')

    compartment_info = CompartmentInfo(list(system_compartments), list(population_partition))

    return compartment_info, past_components


def _split_compartments(compartments: List[str],
                        partition: Dict[str, pd.Series]) -> Set[str]:
    return {f'{compartment}_{partition_group}'
            for compartment, partition_group in itertools.product(compartments, partition)}


def run_normal_ode_model_by_location(initial_condition: pd.DataFrame,
                                     beta_params: Dict[str, float],
                                     betas: pd.DataFrame,
                                     thetas: pd.Series,
                                     scenario_data: ScenarioData,
                                     location_ids: List[int],
                                     scenario_spec: ScenarioSpecification,
                                     compartment_info: CompartmentInfo):
    forecasts = []
    for location_id in location_ids:
        init_cond = initial_condition.loc[location_id].values
        total_population = init_cond.sum()

        model_specs = _SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=total_population,
        )
        parameters = ['beta', 'theta']
        if scenario_spec.system == 'vaccine':
            outcomes = ['unprotected', 'protected', 'immune']
            vaccine_parameters = [f'{outcome}_{group}' for outcome, group in
                                  itertools.product(outcomes, compartment_info.group_suffixes)]
            parameters += vaccine_parameters

        ode_runner = _ODERunner(model_specs, scenario_spec, compartment_info, parameters)

        loc_betas = betas.loc[location_id].sort_values('date')
        loc_days = loc_betas['date']
        loc_times = np.array((loc_days - loc_days.min()).dt.days)
        loc_betas = loc_betas['beta_pred'].values
        loc_thetas = np.repeat(thetas.get(location_id, default=0), loc_betas.size)
        if scenario_data.daily_vaccinations is None:
            loc_vaccinations = np.zeros(shape=loc_betas.shape)
        else:
            loc_vaccinations = scenario_data.daily_vaccinations.loc[location_id]
            loc_vaccinations = loc_vaccinations.merge(loc_days, how='right').sort_values('date')
            loc_vaccinations = loc_vaccinations['daily_vaccinations'].fillna(0).values

        forecasted_components = ode_runner.get_solution(init_cond, loc_times, loc_betas, loc_thetas, loc_vaccinations)
        forecasted_components['date'] = loc_days.values
        forecasted_components['location_id'] = location_id
        forecasts.append(forecasted_components)
    forecasts = pd.concat(forecasts)
    return forecasts


@dataclass(frozen=True)
class _SeiirModelSpecs:
    alpha: float
    sigma: float
    gamma1: float
    gamma2: float
    N: float
    delta: float = 0.1

    def __post_init__(self):
        assert 0 < self.alpha <= 1.0
        assert self.sigma >= 0.0
        assert self.gamma1 >= 0
        assert self.gamma2 >= 0
        assert self.N > 0


class ODESystem:

    def __init__(self,
                 constants: _SeiirModelSpecs,
                 sub_groups: List[str],
                 compartments: List[str],
                 parameters: List[str]):
        self.alpha = constants.alpha
        self.sigma = constants.sigma
        self.gamma1 = constants.gamma1
        self.gamma2 = constants.gamma2
        self.N = constants.N

        self.sub_groups = sub_groups
        self.compartments = compartments
        self.parameters = parameters

    def system(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _SEIIR(ODESystem):

    def _group_system(self, t: float, y: np.ndarray, beta: float, theta_plus: float, theta_minus: float):
        s, e, i1, i2, r = y
        new_e = beta * (s / self.N) * (i1 + i2) ** self.alpha

        ds = -new_e - theta_plus * s
        de = new_e + theta_plus * s - self.sigma * e - theta_minus * e
        di1 = self.sigma * e - self.gamma1 * i1
        di2 = self.gamma1 * i1 - self.gamma2 * i2
        dr = self.gamma2 * i2 + theta_minus * e

        return np.array([ds, de, di1, di2, dr])

    def system(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        beta, theta = p
        theta_plus = max(theta, 0.)
        theta_minus = -min(theta, 0.)

        y = np.split(y.copy(), len(self.sub_groups))
        dy = [self._group_system(t, y_i, beta, theta_plus, theta_minus) for y_i in y]
        dy = np.hstack(dy)

        return dy


class _VaccineSEIIR(ODESystem):

    def _group_system(self, beta: float, theta_plus: float, theta_minus: float,
                      psi: np.array, y: np.array) -> np.array:
        unvaccinated, vaccinated = y[:5], y[5:]
        s, e, i1, i2, r = unvaccinated
        s_v, e_v, i1_v, i2_v, r_v, r_sv = vaccinated

        n_v = sum(unvaccinated)
        i = i1 + i2 + i1_v + i2_v

        psi_tilde = min(psi, n_v) / n_v
        psi_s = min(1 - beta * i**self.alpha / self.N - theta_plus, psi_tilde)
        psi_e = min(1 - self.sigma - theta_minus, psi_tilde)
        psi_i1 = min(1 - self.gamma1, psi_tilde)
        psi_i2 = min(1 - self.gamma2, psi_tilde)
        psi_r = psi_tilde

        new_e = beta * s * i**self.alpha / self.N + theta_plus * s
        ds = -new_e - psi_s*s
        de = new_e - self.sigma*e - theta_minus*e - psi_e*e
        di1 = self.sigma*e - self.gamma1*i1 - psi_i1*i1
        di2 = self.gamma1*i1 - self.gamma2*i2 - psi_i2*i2
        dr = self.gamma2*i2 + theta_minus*e - psi_r*r

        new_e_v = beta * s_v * i**self.alpha / self.N + theta_plus * s_v
        ds_v = -new_e_v + (1-self.eta)*psi_s*s
        de_v = new_e_v - self.sigma*e_v - theta_minus*e_v + psi_e*e
        di1_v = self.sigma*e_v - self.gamma1*i1_v + psi_i1*i1
        di2_v = self.gamma1*i1_v - self.gamma2*i2_v + psi_i2*i2
        dr_v = self.gamma2*i2_v + theta_minus*e_v + psi_r*r
        dr_sv = self.eta*psi_s*s

        return np.array([ds, de, di1, di2, dr,
                         ds_v, de_v, di1_v, di2_v, dr_v, dr_sv])

    def system(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """ODE System.
        """
        beta, theta, *vaccinations = p
        theta_plus = max(theta, 0.)
        theta_minus = -min(theta, 0.)

        infectious = 0
        for compartment, people_in_compartment in zip(self.compartments, y):
            if 'I1' in compartment or 'I2' in compartment:
                infectious += people_in_compartment

        y = np.split(y.copy(), len(self.sub_groups))
        vaccinations = np.split(vaccinations.copy(), len(self.sub_groups))
        #dy = [self._group_system(beta, theta_plus, theta_minus, p si_i, y_i) for psi_i, y_i in zip(psi, y)]
        #dy = np.hstack(dy)

        #return dy


class _ODERunner:

    systems: Dict[str, Type[ODESystem]] = {
        'normal': _SEIIR,
        'vaccine': _VaccineSEIIR
    }
    solvers: Dict[str, Type[ODESolver]] = {
        'RK45': RK4
    }

    def __init__(self,
                 model_specs: _SeiirModelSpecs,
                 scenario_spec: ScenarioSpecification,
                 compartment_info: CompartmentInfo,
                 parameters: List[str]):
        self.model = self.systems[scenario_spec.system](
            model_specs,
            compartment_info.group_suffixes,
            compartment_info.compartments,
            parameters
        )
        self.solver = self.solvers[scenario_spec.solver](self.model.system, model_specs.delta)

    def get_solution(self, initial_condition, times, beta, theta, *scenario_args):
        params = np.vstack((beta, theta, *scenario_args))
        solution = self.solver.solve(
            t=times,
            init_cond=initial_condition,
            t_params=times,
            params=params
        )
        result_array = np.concatenate([
            solution,
            *[x.reshape((1, -1)) for x in [beta, theta, *scenario_args, times]]
        ], axis=0).T
        result = pd.DataFrame(
            data=result_array,
            columns=self.model.compartments + self.model.parameters + ['t']
        )

        return result


def _beta_shift(beta_hat: pd.DataFrame,
                beta_scales: pd.DataFrame) -> pd.DataFrame:
    """Shift the raw predicted beta to line up with beta in the past.

    This method performs both an intercept shift and a scaling based on the
    residuals of the ode fit beta and the beta hat regression in the past.

    Parameters
    ----------
        beta_hat
            Dataframe containing the date, location_id, and beta hat in the
            future.
        beta_scales
            Dataframe containing precomputed parameters for the scaling.

    Returns
    -------
        Predicted beta, after scaling (shift).

    """
    beta_scales = beta_scales.set_index('location_id')
    beta_hat = beta_hat.sort_values(['location_id', 'date']).set_index('location_id')
    scale_init = beta_scales['scale_init']
    scale_final = beta_scales['scale_final']
    window_size = beta_scales['window_size']

    beta_final = []
    for location_id in beta_hat.index.unique():
        if window_size is not None:
            t = np.arange(len(beta_hat.loc[location_id])) / window_size.at[location_id]
            scale = scale_init.at[location_id] + (scale_final.at[location_id] - scale_init.at[location_id]) * t
            scale[(window_size.at[location_id] + 1):] = scale_final.at[location_id]
        else:
            scale = scale_init.at[location_id]
        loc_beta_hat = beta_hat.loc[location_id].set_index('date', append=True)['beta_pred']
        loc_beta_final = loc_beta_hat * scale
        beta_final.append(loc_beta_final)

    beta_final = pd.concat(beta_final).reset_index()

    return beta_final

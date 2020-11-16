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


def prep_seir_parameters(betas: pd.DataFrame,
                         thetas: pd.Series,
                         scenario_data: ScenarioData):
    betas = betas.rename(columns={'beta_pred': 'beta'})
    parameters = betas.merge(thetas, on='location_id')
    if scenario_data.vaccinations is not None:
        parameters = parameters.merge(scenario_data.vaccinations, on=['location_id', 'date'])
    return parameters


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
        partition_map = {}
    elif population_partition == 'high_and_low_risk':
        modeled_locations = population['location_id'].isin(location_ids)
        is_2019 = population['year_id']
        is_both_sexes = population['sex_id'] == 3
        five_year_bins = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31, 32, 235]
        is_five_year_bins = population['age_group_id'].isin(five_year_bins)
        population = population.loc[modeled_locations & is_2019 & is_both_sexes & is_five_year_bins, :]

        total_pop = population.groupby('location_id')['population'].sum()
        low_risk_pop = population[population['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
        high_risk_pop = total_pop - low_risk_pop

        partition_map = {
            'lr': low_risk_pop / total_pop,
            'hr': high_risk_pop / total_pop,
        }
    else:
        raise NotImplementedError

    return partition_map


@dataclass
class CompartmentInfo:
    compartments: List[str]
    group_suffixes: List[str]


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

    if population_partition:
        partitioned_past_components = []
        for compartment in regression_compartments:
            for partition_group, proportion in population_partition.items():
                partitioned_past_components.append(
                    (past_components[compartment] * proportion).rename(f'{compartment}_{partition_group}')
                )
        past_components = pd.concat(partitioned_past_components, axis=1)

    rows_to_fill = past_components.notnull().all(axis=1)
    compartments_to_fill = set(system_compartments).difference(past_components.columns)
    past_components = past_components.reindex(system_compartments, axis=1)
    for compartment_to_fill in compartments_to_fill:
        past_components.loc[rows_to_fill, compartment_to_fill] = 0

    past_components = pd.concat([past_beta, past_components], axis=1).reset_index(level='date')

    compartment_info = CompartmentInfo(list(system_compartments), list(population_partition))

    return compartment_info, past_components


def _split_compartments(compartments: List[str],
                        partition: Dict[str, pd.Series]) -> List[str]:
    # Order of the groupings here matters!
    if not partition:
        return compartments
    return [f'{compartment}_{partition_group}'
            for partition_group, compartment in itertools.product(partition, compartments)]


def run_normal_ode_model_by_location(initial_condition: pd.DataFrame,
                                     beta_params: Dict[str, float],
                                     seir_parameters: pd.DataFrame,
                                     scenario_spec: ScenarioSpecification,
                                     compartment_info: CompartmentInfo):
    forecasts = []

    for location_id, init_cond in initial_condition.iterrows():
        # Index columns to ensure sort order
        init_cond = init_cond[compartment_info.compartments].values
        total_population = init_cond.sum()

        model_specs = _SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=total_population,
        )
        loc_parameters = seir_parameters.loc[location_id].sort_values('date')
        loc_date = loc_parameters['date']
        loc_times = np.array((loc_date - loc_date.min()).dt.days)
        loc_parameters = loc_parameters.set_index('date')

        ode_runner = _ODERunner(model_specs, scenario_spec, compartment_info, loc_parameters.columns.tolist())
        forecasted_components = ode_runner.get_solution(init_cond, loc_times, loc_parameters.values)
        forecasted_components['date'] = loc_date.values
        forecasted_components['location_id'] = location_id
        forecasts.append(forecasted_components)
    forecasts = (pd.concat(forecasts)
                 .drop(columns='t')  # Convenience column in the ode.
                 .set_index(['location_id', 'date'])
                 .reset_index(level='date'))  # Move date out front.
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

        self.sub_groups = sub_groups if sub_groups else ['']
        self.compartments = compartments
        self.parameters_map = {p: i for i, p in enumerate(parameters)}

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
        beta, theta = p[self.parameters_map['beta']], p[self.parameters_map['theta']]
        theta_plus = max(theta, 0.)
        theta_minus = -min(theta, 0.)

        y = np.split(y.copy(), len(self.sub_groups))
        dy = [self._group_system(t, y_i, beta, theta_plus, theta_minus) for y_i in y]
        dy = np.hstack(dy)

        return dy


class _VaccineSEIIR(ODESystem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wasted_vaccines = []

    def _group_system(self, t: float, y: np.array,
                      beta: float, theta_plus: float, theta_minus: float,
                      infectious: float, v_unprotected: float, v_protected: float, v_immune: float) -> np.array:
        # Unpack the compartments
        unvaccinated, unprotected, protected, m = y[:5], y[5:10], y[10:15], y[15]
        s, e, i1, i2, r = unvaccinated
        s_u, e_u, i1_u, i2_u, r_u = unprotected
        s_p, e_p, i1_p, i2_p, r_p = protected

        n = sum(unvaccinated)

        # vaccinated and unprotected come from all bins.
        # Get count coming from S.
        v_unprotected_s = s/n * v_unprotected

        # Expected vaccines coming out of S.
        s_vaccines = v_unprotected_s + v_protected + v_immune
        # Proportion of S vaccines going to each bin.
        if s_vaccines:
            rho_unprotected = v_unprotected_s / s_vaccines
            rho_protected = v_protected / s_vaccines
            rho_immune = v_immune / s_vaccines
        else:
            rho_unprotected, rho_protected, rho_immune = 0, 0, 0
        # Actual count of vaccines coming out of S.
        s_vaccines = min(1 - beta * infectious ** self.alpha / self.N - theta_plus, s_vaccines / s) * s

        # Expected vaccines coming out of E.
        e_vaccines = e/n * v_unprotected
        # Actual vaccines coming out of E.
        e_vaccines = min(1 - self.sigma - theta_minus, e_vaccines / e) * e

        # Expected vaccines coming out of I1.
        i1_vaccines = i1/n * v_unprotected
        # Actual vaccines coming out of I1.
        i1_vaccines = min(1 - self.gamma1, i1_vaccines / i1) * i1

        # Expected vaccines coming out of I2.
        i2_vaccines = i2 / n * v_unprotected
        # Actual vaccines coming out of I2.
        i2_vaccines = min(1 - self.gamma2, i2_vaccines / i2) * i2

        # Expected vaccines coming out of R.
        r_vaccines = r / n * v_unprotected
        # Actual vaccines coming out of R
        r_vaccines = min(1, r_vaccines / r) * r

        # Some vaccine accounting
        expected_vaccines = v_unprotected + v_protected + v_immune
        actual_vaccines = s_vaccines + e_vaccines + i1_vaccines + i2_vaccines + r_vaccines
        self.wasted_vaccines.append(expected_vaccines - actual_vaccines)

        # Unvaccinated equations.
        # Normal Epi + vaccines causing exits from all compartments.
        new_e = beta * s * infectious**self.alpha / self.N + theta_plus * s
        ds = -new_e - s_vaccines
        de = new_e - self.sigma*e - theta_minus*e - e_vaccines
        di1 = self.sigma*e - self.gamma1*i1 - i1_vaccines
        di2 = self.gamma1*i1 - self.gamma2*i2 - i2_vaccines
        dr = self.gamma2*i2 + theta_minus*e - r_vaccines

        # Vaccinated and unprotected equations
        # Normal epi + vaccines causing entrances to all compartments from
        # their unvaccinated counterparts.
        new_e_u = beta * s_u * infectious**self.alpha / self.N + theta_plus * s_u
        ds_u = -new_e_u + rho_unprotected*s_vaccines
        de_u = new_e_u - self.sigma*e_u - theta_minus*e_u + e_vaccines
        di1_u = self.sigma*e_u - self.gamma1*i1_u + i1_vaccines
        di2_u = self.gamma1*i1_u - self.gamma2*i2_u + i2_vaccines
        dr_u = self.gamma2*i2_u + theta_minus*e_u + r_vaccines

        # Vaccinated and protected equations
        # Normal epi + protective vaccines taking people from S and putting
        # them in S_p
        new_e_p = beta * s_p * infectious ** self.alpha / self.N + theta_plus * s_p
        ds_p = -new_e_p + rho_protected*s_vaccines
        de_p = new_e_p - self.sigma * e_p - theta_minus * e_p
        di1_p = self.sigma * e_p - self.gamma1 * i1_p
        di2_p = self.gamma1 * i1_p - self.gamma2 * i2_p
        dr_p = self.gamma2 * i2_p + theta_minus * e_p

        # Vaccinated and immune
        dm = rho_immune*s_vaccines

        return np.array([
            ds, de, di1, di2, dr,
            ds_u, de_u, di1_u, di2_u, dr_u,
            ds_p, de_p, di1_p, di2_p, dr_p,
            dm
        ])

    def system(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """ODE System."""
        beta, theta = p[self.parameters_map['beta']], p[self.parameters_map['theta']]
        theta_plus = max(theta, 0.)
        theta_minus = -min(theta, 0.)
        p_immune = 0.5  # TODO: thread through parameter
        infectious = 0
        for compartment, people_in_compartment in zip(self.compartments, y):
            if 'I1' in compartment or 'I2' in compartment:
                infectious += people_in_compartment

        y = np.split(y.copy(), len(self.sub_groups))
        dy = []
        for group, y_group in zip(self.sub_groups, y):
            unprotected = p[self.parameters_map[f'unprotected_{group}']]
            effective = p[self.parameters_map[f'effectively_vaccinated_{group}']]
            immune, protected = p_immune * effective, (1-p_immune) * effective
            dy.append(
                self._group_system(t, y_group,
                                   beta, theta_plus, theta_minus,
                                   infectious, unprotected, protected, immune)
            )
        
        dy = np.hstack(dy)

        return dy


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

    def get_solution(self, initial_condition, times, parameters):
        solution = self.solver.solve(
            t=times,
            init_cond=initial_condition,
            t_params=times,
            params=parameters.T
        )
        result_array = np.concatenate([
            solution,
            parameters.T,
            times.reshape((1, -1)),
        ], axis=0).T
        result = pd.DataFrame(
            data=result_array,
            columns=self.model.compartments + list(self.model.parameters_map) + ['t']
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

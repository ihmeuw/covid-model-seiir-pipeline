from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING
import itertools

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
    static_vars,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    HospitalCorrectionFactors,
    CompartmentInfo,
    ScenarioData,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.ode_systems import (
    seiir,
    vaccine,
)


if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
        ScenarioSpecification,
    )
    # Support type checking but keep the pipeline stages as isolated as possible.
    from covid_model_seiir_pipeline.pipeline.regression.specification import (
        HospitalParameters,
    )


def forecast_beta(covariates: pd.DataFrame,
                  coefficients: pd.DataFrame,
                  beta_shift_parameters: pd.DataFrame) -> pd.DataFrame:
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    # Rescale the predictions of beta based on the residuals from the
    # regression.
    betas = _beta_shift(beta_hat, beta_shift_parameters).set_index('location_id')
    return betas


def forecast_correction_factors(correction_factors: HospitalCorrectionFactors,
                                today: pd.Series,
                                max_date: pd.Timestamp,
                                hospital_parameters: 'HospitalParameters') -> HospitalCorrectionFactors:
    averaging_window = pd.Timedelta(days=hospital_parameters.correction_factor_average_window)
    application_window = pd.Timedelta(days=hospital_parameters.correction_factor_application_window)
    assert np.all(max_date > today + application_window)

    new_cfs = {}
    for cf_name, cf in utilities.asdict(correction_factors).items():
        loc_cfs = []
        for loc_id in today.index:
            loc_cf = cf.loc[loc_id]
            loc_today = today.loc[loc_id]
            mean_cf = loc_cf.loc[loc_today - averaging_window: loc_today].mean()
            loc_cf = loc_cf.loc[:loc_today]
            loc_cf.loc[loc_today + application_window] = mean_cf
            loc_cf.loc[max_date] = mean_cf
            loc_cf = loc_cf.asfreq('D').interpolate().reset_index()
            loc_cf['location_id'] = loc_id
            loc_cfs.append(loc_cf.set_index(['location_id', 'date'])[cf_name])
        new_cfs[cf_name] = pd.concat(loc_cfs).sort_index()
    return HospitalCorrectionFactors(**new_cfs)


def correct_ifr(ifr: pd.Series, variant_scalar: pd.Series, forecast_end_date: pd.Timestamp):
    ifr = (ifr
           .reindex(ifr.index.union(variant_scalar.index))
           .groupby('location_id')
           .fillna(method='ffill'))
    ifr = ifr.loc[pd.IndexSlice[:, :forecast_end_date]]
    ifr.loc[variant_scalar.index] *= variant_scalar
    return ifr


def prep_seiir_parameters(betas: pd.DataFrame,
                          thetas: pd.Series,
                          scenario_data: ScenarioData):
    betas = betas.rename(columns={'beta_pred': 'beta'})
    parameters = betas.merge(thetas, on='location_id')
    if scenario_data.vaccinations is not None:
        v = scenario_data.vaccinations
        parameters = parameters.merge(v, on=['location_id', 'date'], how='left').fillna(0)
    return parameters


def get_population_partition(population: pd.DataFrame,
                             population_partition: str) -> Dict[str, pd.Series]:
    """Create a location-specific partition of the population.

    Parameters
    ----------
    population
        A dataframe with location, age, and sex specific populations.
    population_partition
        A string describing how the population should be partitioned.

    Returns
    -------
        A mapping between the SEIIR compartment suffix for the partition groups
        and a series mapping location ids to the proportion of people in each
        compartment that should be allocated to the partition group.

    """
    if population_partition == 'none':
        partition_map = {}
    elif population_partition == 'high_and_low_risk':
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


def get_past_components(beta_regression_df: pd.DataFrame,
                        population_partition: Dict[str, pd.Series],
                        ode_system: str) -> Tuple[CompartmentInfo, pd.DataFrame]:
    regression_compartments = static_vars.SEIIR_COMPARTMENTS
    system_compartment_map = {
        'normal': _split_compartments(static_vars.SEIIR_COMPARTMENTS, population_partition),
        'vaccine': _split_compartments(static_vars.VACCINE_SEIIR_COMPARTMENTS, population_partition),
    }
    system_compartments = system_compartment_map[ode_system]

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
                                     seiir_parameters: pd.DataFrame,
                                     scenario_spec: 'ScenarioSpecification',
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
            system_params=scenario_spec.system_params.copy(),
        )
        loc_parameters = seiir_parameters.loc[location_id].sort_values('date')
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
    system_params: dict

    def __post_init__(self):
        assert 0 < self.alpha <= 1.0
        assert self.sigma >= 0.0
        assert self.gamma1 >= 0
        assert self.gamma2 >= 0
        assert self.N > 0


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


class _ODERunner:
    systems: Dict[str, Callable] = {
        'normal': seiir.system,
        'vaccine': vaccine.system,
    }

    def __init__(self,
                 model_specs: _SeiirModelSpecs,
                 scenario_spec: 'ScenarioSpecification',
                 compartment_info: CompartmentInfo,
                 parameters: List[str]):
        self.system = self.systems[scenario_spec.system]
        self.model_specs = model_specs
        self.scenario_spec = scenario_spec
        self.compartment_info = compartment_info
        self.parameters_map = {p: i for i, p in enumerate(parameters)}

    def get_solution(self, initial_condition, times, parameters):
        parameters = parameters.T  # Each row is a param, each column a day
        # Add the time invariant constants up front.
        constants = [
            self.model_specs.alpha * np.ones_like(parameters[0]),
            self.model_specs.sigma * np.ones_like(parameters[0]),
            self.model_specs.gamma1 * np.ones_like(parameters[0]),
            self.model_specs.gamma2 * np.ones_like(parameters[0]),
        ]
        system_params = [
            parameters[self.parameters_map['beta']],
            np.maximum(parameters[self.parameters_map['theta']], 0),  # Theta plus
            -np.minimum(parameters[self.parameters_map['theta']], 0),  # Theta minus
        ]
        if self.scenario_spec.system == 'vaccine':
            constants.append(
                self.model_specs.system_params.get('proportion_immune', 0.5) * np.ones_like(parameters[0])
            )
            for risk_group in self.compartment_info.group_suffixes:
                system_params.append(parameters[self.parameters_map[f'unprotected_{risk_group}']])
                system_params.append(parameters[self.parameters_map[f'protected_{risk_group}']])
                system_params.append(parameters[self.parameters_map[f'immune_{risk_group}']])

        system_params = np.vstack([
            constants,
            system_params,
        ])

        solution = math.solve_ode(
            system=self.system,
            t=times,
            init_cond=initial_condition,
            params=system_params,
        )

        result_array = np.concatenate([
            solution,
            parameters,
            times.reshape((1, -1)),
        ], axis=0).T
        result = pd.DataFrame(
            data=result_array,
            columns=self.compartment_info.compartments + list(self.parameters_map) + ['t'],
        )

        return result

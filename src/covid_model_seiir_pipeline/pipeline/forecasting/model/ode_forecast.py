from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING
import itertools

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    ModelParameters,
    InitialCondition,
    CompartmentInfo,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.ode_systems import (
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


##############################
# ODE parameter construction #
##############################

def build_model_parameters(indices: Indices,
                           ode_parameters: pd.DataFrame,
                           beta_regression: pd.DataFrame,
                           thetas: pd.Series,
                           covariates: pd.DataFrame,
                           coefficients: pd.DataFrame,
                           beta_scales: pd.DataFrame,
                           vaccine_data: pd.DataFrame,
                           scenario_spec: 'ScenarioSpecification') -> ModelParameters:
    # These are all the same by draw.  Just broadcasting them over a new index.
    alpha = pd.Series(ode_parameters.alpha.mean(), index=indices.full, name='alpha')
    sigma = pd.Series(ode_parameters.sigma.mean(), index=indices.full, name='sigma')
    gamma1 = pd.Series(ode_parameters.gamma1.mean(), index=indices.full, name='gamma1')
    gamma2 = pd.Series(ode_parameters.gamma2.mean(), index=indices.full, name='gamma2')

    beta, beta_wild, beta_variant, p_wild, p_variant, p_all_variant = get_betas_and_prevalences(
        indices,
        beta_regression,
        covariates,
        coefficients,
        beta_scales,
        scenario_spec.variant_beta_scale,
    )

    thetas = thetas.reindex(indices.full, level='location_id')

    if ((1 < thetas) | thetas < -1).any():
        raise ValueError('Theta must be between -1 and 1.')
    if (sigma - thetas >= 1).any():
        raise ValueError('Sigma - theta must be smaller than 1')

    theta_plus = np.maximum(thetas, 0).rename('theta_plus')
    theta_minus = -np.minimum(thetas, 0).rename('theta_minus')

    vaccine_data = vaccine_data.reindex(indices.full, fill_value=0)
    adjusted_vaccinations = adjust_vaccinations_for_variants(vaccine_data, covariates, scenario_spec.system)

    probability_cross_immune = pd.Series(scenario_spec.probability_cross_immune,
                                         index=indices.full, name='probability_cross_immune')

    return ModelParameters(
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        gamma1=gamma1,
        gamma2=gamma2,
        theta_plus=theta_plus,
        theta_minus=theta_minus,
        **adjusted_vaccinations,
        beta_wild=beta_wild,
        beta_variant=beta_variant,
        p_wild=p_wild,
        p_variant=p_variant,
        p_all_variant=p_all_variant,
        probability_cross_immune=probability_cross_immune,
    )


def adjust_vaccinations_for_variants(vaccine_data: pd.DataFrame, covariates: pd.DataFrame, system: str):
    if 'variant' in system:
        return _adjust_variant_vaccines(vaccine_data)
    else:
        return _adjust_non_variant_vaccines(vaccine_data, covariates)


def _adjust_variant_vaccines(vaccine_data):
    risk_groups = ['lr', 'hr']
    vaccinations = {}

    for risk_group in risk_groups:
        base_col_map = {
            f'unprotected_{risk_group}': f'unprotected_{risk_group}',
            f'protected_wild_type_{risk_group}': f'effective_protected_wildtype_{risk_group}',
            f'protected_all_types_{risk_group}': f'effective_protected_variant_{risk_group}',
            f'immune_wild_type_{risk_group}': f'effective_wildtype_{risk_group}',
            f'immune_all_types_{risk_group}': f'effective_variant_{risk_group}',
        }
        for to_name, from_name in base_col_map.items():
            vaccinations[to_name] = vaccine_data[from_name].rename(to_name)
    return vaccinations


def _adjust_non_variant_vaccines(vaccine_data, covariates):
    bad_variant_prevalence = covariates[['variant_prevalence_B1351', 'variant_prevalence_P1']].sum(axis=1)
    variant_start_threshold = pd.Timestamp('2021-05-01')
    location_ids = vaccine_data.reset_index().location_id.tolist()
    bad_variant_entrance_date = (bad_variant_prevalence[bad_variant_prevalence > 1]
                                 .reset_index()
                                 .groupby('location_id')
                                 .date
                                 .min())
    locs_with_bad_variant = (bad_variant_entrance_date[bad_variant_entrance_date < variant_start_threshold]
                             .reset_index()
                             .location_id
                             .tolist())
    locs_without_bad_variant = list(set(location_ids).difference(locs_with_bad_variant))

    risk_groups = ['lr', 'hr']
    vaccinations = {}
    for risk_group in risk_groups:
        bad_variant_col_map = {
            f'unprotected_{risk_group}': [f'unprotected_{risk_group}',
                                          f'effective_protected_wildtype_{risk_group}',
                                          f'effective_wildtype_{risk_group}'],
            f'protected_wild_type_{risk_group}': [],
            f'protected_all_types_{risk_group}': [f'effective_protected_variant_{risk_group}'],
            f'immune_wild_type_{risk_group}': [],
            f'immune_all_types_{risk_group}': [f'effective_variant_{risk_group}'],
        }
        not_bad_variant_col_map = {
            f'unprotected_{risk_group}': [f'unprotected_{risk_group}'],
            f'protected_wild_type_{risk_group}': [],
            f'protected_all_types_{risk_group}': [f'effective_protected_wildtype_{risk_group}',
                                                  f'effective_protected_variant_{risk_group}'],
            f'immune_wild_type_{risk_group}': [],
            f'immune_all_types_{risk_group}': [f'effective_wildtype_{risk_group}',
                                               f'effective_variant_{risk_group}'],
        }
        bad_variant_vaccines = {
            name: vaccine_data[cols].sum(axis=1).rename(name) for name, cols in bad_variant_col_map.items()
        }
        not_bad_variant_vaccines = {
            name: vaccine_data[cols].sum(axis=1).rename(name) for name, cols in not_bad_variant_col_map.items()
        }
        for name in bad_variant_vaccines:
            vaccinations[name] = (bad_variant_vaccines[name]
                                  .loc[locs_with_bad_variant]
                                  .append(not_bad_variant_vaccines[name].loc[locs_without_bad_variant])
                                  .sort_index())
    return vaccinations


def get_betas_and_prevalences(indices: Indices,
                              beta_regression: pd.DataFrame,
                              covariates: pd.DataFrame,
                              coefficients: pd.DataFrame,
                              beta_shift_parameters: pd.DataFrame,
                              variant_beta_scale: float) -> Tuple[pd.Series, pd.Series, pd.Series,
                                                                  pd.Series, pd.Series, pd.Series]:
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).loc[indices.future].rename('beta_hat').reset_index()
    beta = (beta_shift(beta_hat, beta_shift_parameters)
            .set_index(['location_id', 'date'])
            .beta_hat
            .rename('beta'))
    beta = beta_regression.loc[indices.past, 'beta'].append(beta)
    log_beta = np.log(beta)

    coef = {}
    prev = {}
    effects = {}
    for v in ['B117', 'B1351', 'P1']:
        coef[v] = coefficients[f'variant_prevalence_{v}']
        prev[v] = covariates[f'variant_prevalence_{v}'].fillna(0)
        effects[v] = coef[v] * prev[v]

    p_variant = (prev['B1351'] + prev['P1']).rename('p_variant')
    p_wild = (1 - p_variant).rename('p_wild')
    p_w = 1 - sum(prev.values())

    log_beta_w = log_beta - sum(effects.values())
    beta_w = np.exp(log_beta_w)
    beta_b117 = np.exp(log_beta_w + coef['B117'])

    beta_wild = (((p_w * beta_w + prev['B117'] * beta_b117) / p_wild)
                 .groupby('location_id')
                 .fillna(method='bfill')
                 .rename('beta_wild'))
    beta_variant = (beta_w + variant_beta_scale * (beta_b117 - beta_w)).rename('beta_variant')

    p_all_variant = sum(prev.values()).rename('p_all_variant')

    return beta, beta_wild, beta_variant, p_wild, p_variant, p_all_variant


def beta_shift(beta_hat: pd.DataFrame,
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
        loc_beta_hat = beta_hat.loc[location_id].set_index('date', append=True)['beta_hat']
        loc_beta_final = loc_beta_hat * scale
        beta_final.append(loc_beta_final)

    beta_final = pd.concat(beta_final).reset_index()

    return beta_final


###################################
# Past compartment redistribution #
###################################

def redistribute_past_compartments(indices: Indices,
                                   compartments: pd.DataFrame,
                                   population: pd.DataFrame,
                                   model_parameters: ModelParameters):
    import pdb; pdb.set_trace()
    pass








def get_component_groups(model_parameters: ModelParameters):
    new_e = infection_data.loc[:, 'infections'].groupby('location_id').fillna(0)
    total_pop = population.groupby('location_id')['population'].sum()
    low_risk_pop = population[population['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
    high_risk_pop = total_pop - low_risk_pop
    pop_weights = {
        'lr': low_risk_pop / total_pop,
        'hr': high_risk_pop / total_pop,
    }
    simple_comp_diff = simple_comp.groupby('location_id').diff()

    # FIXME: These both need some real attention.
    vaccine_columns = [f'{c}_{g}' for g, c in itertools.product(pop_weights, vaccine.COMPARTMENTS)]
    vaccine_comp_diff = pd.DataFrame(data=0., columns=vaccine_columns, index=simple_comp.index)
    for risk_group, pop_weight in pop_weights.items():
        for column in seiir.COMPARTMENTS:
            vaccine_comp_diff[f'{column}_{risk_group}'] = simple_comp_diff[column] * pop_weight
    vaccine_comp = vaccine_comp_diff.groupby('location_id').cumsum()
    vaccine_comp['S_lr'] += low_risk_pop
    vaccine_comp['S_hr'] += high_risk_pop

    variant_columns = [f'{c}_{g}' for g, c in itertools.product(pop_weights, variant.COMPARTMENTS)]
    variant_comp_diff = pd.DataFrame(data=0., columns=variant_columns, index=simple_comp.index)
    variant_prevalence = model_parameters.p_variant.loc[simple_comp.index]
    prob_cross_immune = model_parameters.probability_cross_immune.loc[simple_comp.index]
    for risk_group, pop_weight in pop_weights.items():
        # Just split S by demography.
        variant_comp_diff[f'S_{risk_group}'] = simple_comp_diff['S'] * pop_weight
        # E, I1, and I2 are fast enough to move through that just using the
        # variant prevalence is not a bad estimation of initial condition.
        for compartment in ['E', 'I1', 'I2']:
            variant_comp_diff[f'{compartment}_{risk_group}'] = (
                simple_comp_diff[compartment] * pop_weight * (1 - variant_prevalence)
            )
            variant_comp_diff[f'{compartment}_variant_{risk_group}'] = (
                simple_comp_diff[compartment] * pop_weight * variant_prevalence
            )
        # Who's in R vs. S_variant depends roughly on the probability of cross immunity.
        # This is a bad approximation if variant prevalence is high and there have been a significant
        # of infections.
        variant_comp_diff[f'S_variant_{risk_group}'] = simple_comp_diff['R'] * pop_weight * (1 - prob_cross_immune)
        variant_comp_diff[f'R_{risk_group}'] = simple_comp_diff['R'] * pop_weight * prob_cross_immune

        variant_comp_diff[f'NewE_wild_{risk_group}'] = new_e * pop_weight * (1 - variant_prevalence)
        variant_comp_diff[f'NewE_variant_{risk_group}'] = new_e * pop_weight * variant_prevalence
    variant_comp = variant_comp_diff.groupby('location_id').cumsum()
    variant_comp['S_lr'] += low_risk_pop
    variant_comp['S_hr'] += high_risk_pop

    return vaccine_comp.loc[index], variant_comp.loc[index]



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


class _ODERunner:
    systems: Dict[str, Callable] = {
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

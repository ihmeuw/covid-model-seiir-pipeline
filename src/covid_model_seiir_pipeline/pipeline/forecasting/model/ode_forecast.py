from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    ModelParameters,
    InitialCondition,
    PostprocessingParameters,
    RatioData,
    HospitalCorrectionFactors,
    CompartmentInfo,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.ode_systems import (
    seiir,
    vaccine,
    variant,
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
                           beta_params: Dict[str, float],
                           thetas: pd.Series,
                           covariates: pd.DataFrame,
                           coefficients: pd.DataFrame,
                           beta_scales: pd.DataFrame,
                           vaccine_data: pd.DataFrame,
                           scenario_spec: 'ScenarioSpecification') -> ModelParameters:
    alpha = pd.Series(beta_params['alpha'], index=indices.full, name='alpha')
    sigma = pd.Series(beta_params['sigma'], index=indices.full, name='sigma')
    gamma1 = pd.Series(beta_params['gamma1'], index=indices.full, name='gamma1')
    gamma2 = pd.Series(beta_params['gamma2'], index=indices.full, name='gamma2')

    beta, beta_b117, beta_b1351 = forecast_beta(covariates, coefficients, beta_scales)

    thetas = thetas.reindex(indices.full, level='location_id')

    if ((1 < thetas) | thetas < -1).any():
        raise ValueError('Theta must be between -1 and 1.')
    if (sigma - thetas >= 1).any():
        raise ValueError('Sigma - theta must be smaller than 1')

    theta_plus = np.maximum(thetas, 0)
    theta_minus = -np.minimum(thetas, 0)

    vaccine_data = vaccine_data.reindex(indices.full, fill_value=0)
    adjusted_vaccinations = adjust_vaccinations_for_variants(vaccine_data, covariates)

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
        beta_b117=beta_b117,
        beta_b1351=beta_b1351,
        b117_prevalence=covariates['variant_prevalence_B117'],
        b1351_prevalence=covariates['variant_prevalence_B1351'],
        probability_cross_immune=probability_cross_immune,
    )


def adjust_vaccinations_for_variants(vaccine_data: pd.DataFrame, covariates: pd.DataFrame):
    vaccine_data = vaccine_data.reindex(covariates.index, fill_value=0)
    if 'variant_prevalence_B1351' in covariates.columns:
        b1351_prevalence = covariates['variant_prevalence_B1351']
    else:
        b1351_prevalence = pd.Series(0, index=vaccine_data.index)

    max_prevalence = b1351_prevalence.groupby('location_id').max()
    locs_with_b1351 = max_prevalence[max_prevalence > 0].index.tolist()
    locs_without_b1351 = max_prevalence.index.difference(locs_with_b1351).tolist()

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

        b1351_col_map = {
            f'old_unprotected_{risk_group}': [f'unprotected_{risk_group}',
                                              f'effective_protected_wildtype_{risk_group}',
                                              f'effective_wildtype_{risk_group}'],
            f'old_protected_{risk_group}': [f'effective_protected_variant_{risk_group}'],
            f'old_immune_{risk_group}': [f'effective_variant_{risk_group}'],
        }
        not_b1351_col_map = {
            f'old_unprotected_{risk_group}': [f'unprotected_{risk_group}'],
            f'old_protected_{risk_group}': [f'effective_protected_wildtype_{risk_group}',
                                            f'effective_protected_variant_{risk_group}'],
            f'old_immune_{risk_group}': [f'effective_wildtype_{risk_group}',
                                         f'effective_variant_{risk_group}'],
        }
        b1351_vaccines = {name: vaccine_data[cols].sum(axis=1).rename(name) for name, cols in b1351_col_map.items()}
        not_b1351_vaccines = {name: vaccine_data[cols].sum(axis=1).rename(name)
                              for name, cols in not_b1351_col_map.items()}
        for name in b1351_vaccines:
            vaccinations[name] = (
                b1351_vaccines[name]
                .loc[locs_with_b1351]
                .append(
                    not_b1351_vaccines[name]
                    .loc[locs_without_b1351]
                ).sort_index()
            )
    return vaccinations


def forecast_beta(covariates: pd.DataFrame,
                  coefficients: pd.DataFrame,
                  beta_shift_parameters: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    drop_cols = {'beta': ['variant_prevalence_B117', 'variant_prevalence_B1351'],
                 'beta_b117': ['variant_prevalence_B1351'],
                 'beta_b1351': ['variant_prevalence_B117']}

    betas = {}
    for beta_name, drop in drop_cols.items():

        log_beta_hat = math.compute_beta_hat(
            covariates.drop(columns=drop).reset_index(),
            coefficients.drop(columns=drop).reset_index(),
        )
        beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

        beta = (beta_shift(beta_hat, beta_shift_parameters)
                .set_index(['location_id', 'date'])
                .beta_pred
                .rename(beta_name))
        betas[beta_name] = beta

    return betas['beta'], betas['beta_b117'], betas['beta_b1351']


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


##################################
# Initial condition construction #
##################################

def build_initial_condition(indices: Indices,
                            beta_regression: pd.DataFrame,
                            population: pd.DataFrame) -> InitialCondition:
    simple_ic, vaccine_ic, variant_ic = get_component_groups(beta_regression, population, indices.initial_condition)
    # Date column has served its purpose.  ODE only cares about t0, not what it is.
    return InitialCondition(
        simple=simple_ic,
        vaccine=vaccine_ic,
        variant=variant_ic,
    )


def get_component_groups(beta_regression_df, population, index):
    simple_ic = beta_regression_df.loc[index, seiir.COMPARTMENTS].reset_index(level='date', drop=True)

    risk_groups = ['lr', 'hr']
    total_pop = population.groupby('location_id')['population'].sum()
    low_risk_pop = population[population['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
    high_risk_pop = total_pop - low_risk_pop

    # FIXME: These both need some real attention.
    vaccine_columns = []
    for risk_group in risk_groups:
        for vaccine_compartment in vaccine.COMPARTMENTS:
            vaccine_columns.append(f'{vaccine_compartment}_{risk_group}')
    vaccine_ic = pd.DataFrame(data=0., columns=vaccine_columns, index=index)
    for column in seiir.COMPARTMENTS:
        vaccine_ic[f'{column}_lr'] = simple_ic[column] * low_risk_pop / total_pop
        vaccine_ic[f'{column}_hr'] = simple_ic[column] * high_risk_pop / total_pop

    variant_columns = []
    for risk_group in risk_groups:
        for variant_compartment in variant.COMPARTMENTS:
            variant_columns.append(f'{variant_compartment}_{risk_group}')
    variant_ic = pd.DataFrame(data=0., columns=variant_columns, index=index)
    for column in seiir.COMPARTMENTS:
        variant_ic[f'{column}_lr'] = simple_ic[column] * low_risk_pop / total_pop
        variant_ic[f'{column}_hr'] = simple_ic[column] * high_risk_pop / total_pop

    return simple_ic, vaccine_ic, variant_ic


#######################################
# Construct postprocessing parameters #
#######################################


def build_postprocessing_parameters(indices: Indices,
                                    beta_regression: pd.DataFrame,
                                    infection_data: pd.DataFrame,
                                    population: pd.DataFrame,
                                    ratio_data: RatioData,
                                    model_parameters: ModelParameters,
                                    correction_factors: HospitalCorrectionFactors,
                                    hospital_parameters: 'HospitalParameters',
                                    scenario_spec: 'ScenarioSpecification') -> PostprocessingParameters:
    beta, compartments = build_past_compartments(
        indices.past,
        beta_regression,
        population,
        scenario_spec.system,
    )
    ratio_data = correct_ratio_data(indices, ratio_data, model_parameters, scenario_spec.variant_ifr_scale)

    correction_factors = forecast_correction_factors(
        indices,
        correction_factors,
        hospital_parameters,
    )

    return PostprocessingParameters(
        past_beta=beta,
        past_compartments=compartments,
        past_infections=infection_data.loc[indices.past, 'infections'],
        past_deaths=infection_data.loc[indices.past, 'deaths'],
        **ratio_data.to_dict(),
        **correction_factors.to_dict()
    )


def build_past_compartments(index: pd.MultiIndex,
                            beta_regression: pd.DataFrame,
                            population: pd.DataFrame,
                            system: str) -> Tuple[pd.Series, pd.DataFrame]:
    simple_comp, vaccine_comp, variant_comp = get_component_groups(beta_regression, population, index)
    if system == 'normal':
        compartments = simple_comp
    elif system == 'vaccine':
        compartments = vaccine_comp
    else:
        compartments = variant_comp
    beta = beta_regression.loc[index, 'beta']
    return beta, compartments


def correct_ratio_data(indices: Indices,
                       ratio_data: RatioData,
                       model_params: ModelParameters,
                       ifr_scale: float) -> RatioData:
    variant_prevalence = model_params.b117_prevalence + model_params.b1351_prevalence
    ifr_scalar = ifr_scale * variant_prevalence + (1 - variant_prevalence)

    ratio_data.ifr = ratio_data.ifr.reindex(indices.full, method='ffill') * ifr_scalar
    ratio_data.ifr_lr = ratio_data.ifr_lr.reindex(indices.full, method='ffill') * ifr_scalar
    ratio_data.ifr_hr = ratio_data.ifr_hr.reindex(indices.full, method='ffill') * ifr_scalar

    ratio_data.idr = ratio_data.idr.reindex(indices.full, method='ffill')
    ratio_data.ihr = ratio_data.ihr.reindex(indices.full, method='ffill')
    return ratio_data


def forecast_correction_factors(indices: Indices,
                                correction_factors: HospitalCorrectionFactors,
                                hospital_parameters: 'HospitalParameters') -> HospitalCorrectionFactors:
    averaging_window = pd.Timedelta(days=hospital_parameters.correction_factor_average_window)
    application_window = pd.Timedelta(days=hospital_parameters.correction_factor_application_window)

    new_cfs = {}
    for cf_name, cf in correction_factors.to_dict().items():
        cf = cf.reindex(indices.full)
        loc_cfs = []
        for loc_id, loc_today in indices.initial_condition.tolist():
            loc_cf = cf.loc[loc_id]
            mean_cf = loc_cf.loc[loc_today - averaging_window: loc_today].mean()
            loc_cf.loc[loc_today:] = np.nan
            loc_cf.loc[loc_today + application_window:] = mean_cf
            loc_cf = loc_cf.interpolate().reset_index()
            loc_cf['location_id'] = loc_id
            loc_cfs.append(loc_cf.set_index(['location_id', 'date'])[cf_name])
        new_cfs[cf_name] = pd.concat(loc_cfs).sort_index()
    return HospitalCorrectionFactors(**new_cfs)


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
    delta: float = 0.1

    def __post_init__(self):
        assert 0 < self.alpha <= 1.0
        assert self.sigma >= 0.0
        assert self.gamma1 >= 0
        assert self.gamma2 >= 0
        assert self.N > 0





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
            dt=self.model_specs.delta,
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

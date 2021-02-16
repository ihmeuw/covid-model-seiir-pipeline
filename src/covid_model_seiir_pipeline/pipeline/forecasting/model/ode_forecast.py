from typing import Callable, Dict, List, NamedTuple, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import tqdm

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

    beta, beta_wild, beta_b117, beta_b1351, beta_p1 = forecast_beta(covariates, coefficients, beta_scales)

    thetas = thetas.reindex(indices.full, level='location_id')

    if ((1 < thetas) | thetas < -1).any():
        raise ValueError('Theta must be between -1 and 1.')
    if (sigma - thetas >= 1).any():
        raise ValueError('Sigma - theta must be smaller than 1')

    theta_plus = np.maximum(thetas, 0)
    theta_minus = -np.minimum(thetas, 0)

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
        beta_b117=beta_b117,
        beta_b1351=beta_b1351,
        beta_p1=beta_p1,
        b117_prevalence=covariates['variant_prevalence_B117'],
        b1351_prevalence=covariates['variant_prevalence_B1351'],
        p1_prevalence=covariates['variant_prevalence_P1'],
        probability_cross_immune=probability_cross_immune,
    )


def adjust_vaccinations_for_variants(vaccine_data: pd.DataFrame, covariates: pd.DataFrame, system: str):
    risk_groups = ['lr', 'hr']
    vaccinations = {}

    if 'variant' in system:
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
    else:
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
                vaccinations[name] = (
                    bad_variant_vaccines[name]
                    .loc[locs_with_bad_variant]
                    .append(
                        not_bad_variant_vaccines[name]
                        .loc[locs_without_bad_variant]
                    ).sort_index()
                )
    return vaccinations


def forecast_beta(covariates: pd.DataFrame,
                  coefficients: pd.DataFrame,
                  beta_shift_parameters: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    drop_cols = {
        'beta': [],
        'beta_wild': ['variant_prevalence_B117', 'variant_prevalence_B1351', 'variant_prevalence_P1'],
        'beta_b117': ['variant_prevalence_B1351', 'variant_prevalence_P1'],
        'beta_b1351': ['variant_prevalence_B117', 'variant_prevalence_P1'],
        'beta_p1': ['variant_prevalence_B117', 'variant_prevalence_B1351'],
    }

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

    return betas['beta'], betas['beta_wild'], betas['beta_b117'], betas['beta_b1351'], betas['beta_p1']


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
        loc_beta_hat = beta_hat.loc[location_id].set_index('date', append=True)['beta_pred']
        loc_beta_final = loc_beta_hat * scale
        beta_final.append(loc_beta_final)

    beta_final = pd.concat(beta_final).reset_index()

    return beta_final


##################################
# Initial condition construction #
##################################

def build_initial_condition(indices: Indices,
                            model_parameters: ModelParameters,
                            beta_regression: pd.DataFrame,
                            infection_data: pd.DataFrame,
                            population: pd.DataFrame) -> InitialCondition:
    simple_ic, vaccine_ic, variant_ic = get_component_groups(
        model_parameters,
        beta_regression,
        infection_data,
        population,
        indices.initial_condition
    )
    # Date column has served its purpose.  ODE only cares about t0, not what it is.
    return InitialCondition(
        simple=simple_ic.reset_index(level='date', drop=True),
        vaccine=vaccine_ic.reset_index(level='date', drop=True),
        variant=variant_ic.reset_index(level='date', drop=True),
    )


def get_component_groups(model_parameters: ModelParameters,
                         beta_regression_df: pd.DataFrame,
                         infection_data: pd.DataFrame,
                         population: pd.DataFrame,
                         index: pd.MultiIndex):
    simple_comp = beta_regression_df.loc[index, seiir.COMPARTMENTS]
    new_e = infection_data.loc[index]

    total_pop = population.groupby('location_id')['population'].sum()
    low_risk_pop = population[population['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
    high_risk_pop = total_pop - low_risk_pop
    pop_weights = {
        'lr': low_risk_pop / total_pop,
        'hr': high_risk_pop / total_pop,
    }
    simple_comp_diff = simple_comp.diff()

    # FIXME: These both need some real attention.
    vaccine_columns = [f'{c}_{g}' for c, g in zip(vaccine.COMPARTMENTS, pop_weights)]
    vaccine_comp_diff = pd.DataFrame(data=0., columns=vaccine_columns, index=index)
    for risk_group, pop_weight in pop_weights.items():
        for column in seiir.COMPARTMENTS:
            vaccine_comp_diff[f'{column}_{risk_group}'] = simple_comp_diff[column] * pop_weight
    vaccine_comp = vaccine_comp_diff.cumsum()

    variant_columns = [f'{c}_{g}' for c, g in zip(variant.COMPARTMENTS, pop_weights)]
    variant_comp_diff = pd.DataFrame(data=0., columns=variant_columns, index=index)
    variant_prevalence = (model_parameters.b1351_prevalence + model_parameters.p1_prevalence).loc[index]
    prob_cross_immune = model_parameters.probability_cross_immune.loc[index]
    for risk_group, pop_weight in pop_weights.items():
        # Just split S by demography.
        variant_comp_diff[f'S_{risk_group}'] = simple_comp_diff['S'] * pop_weight
        # E, I1, and I2 are fast enough to move through that just using the
        # variant prevalence is not a bad estimation of initial condition.
        for compartment in ['E', 'I1', 'I2']:
            variant_comp_diff[f'{compartment}_{risk_group}'] = (
                simple_comp_diff[f'{compartment}_{risk_group}'] * pop_weight * (1 - variant_prevalence)
            )
            variant_comp_diff[f'{compartment}_variant_{risk_group}'] = (
                    simple_comp_diff[f'{compartment}_{risk_group}'] * pop_weight * variant_prevalence
            )
        # Who's in R vs. S_variant depends roughly on the probability of cross immunity.
        # This is a bad approximation if variant prevalence is high and there have been a significant
        # of infections.
        variant_comp_diff[f'S_variant_{risk_group}'] = simple_comp_diff['R'] * pop_weight * (1 - prob_cross_immune)
        variant_comp_diff[f'R_{risk_group}'] = simple_comp_diff['R'] * pop_weight * (1 - prob_cross_immune)

        variant_comp_diff[f'NewE_wild_{risk_group}'] = new_e * pop_weight * (1 - variant_prevalence)
        variant_comp_diff[f'NewE_variant_{risk_group}'] = new_e * pop_weight * variant_prevalence

    variant_comp = variant_comp_diff.cumsum()

    return simple_comp, vaccine_comp, variant_comp


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
        model_parameters,
        beta_regression,
        infection_data,
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
                            model_parameters: ModelParameters,
                            beta_regression: pd.DataFrame,
                            infection_data: pd.DataFrame,
                            population: pd.DataFrame,
                            system: str) -> Tuple[pd.Series, pd.DataFrame]:
    simple_comp, vaccine_comp, variant_comp = get_component_groups(
        model_parameters,
        beta_regression,
        infection_data,
        population,
        index
    )
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

    ratio_data.ifr = ifr_scalar * _expand_rate(ratio_data.ifr, indices.full)
    ratio_data.ifr_lr = ifr_scalar * _expand_rate(ratio_data.ifr_lr, indices.full)
    ratio_data.ifr_hr = ifr_scalar * _expand_rate(ratio_data.ifr_lr, indices.full)

    ratio_data.idr = _expand_rate(ratio_data.idr, indices.full)
    ratio_data.ihr = _expand_rate(ratio_data.ihr, indices.full)
    return ratio_data


def _expand_rate(rate: pd.Series, index: pd.MultiIndex):
    return (rate
            .reindex(index)
            .groupby('location_id')
            .fillna(method='ffill')
            .fillna(method='bfill'))


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


###########
# Run ODE #
###########


class _ODESystem(NamedTuple):
    system: Callable
    initial_condition: pd.DataFrame
    parameters: List[pd.Series]


def run_ode_model(initial_condition: InitialCondition,
                  model_parameters: ModelParameters,
                  system_type: str,
                  progress_bar: bool,
                  dt: float = 0.1):
    systems = {
        'normal': _ODESystem(
            system=seiir.system,
            initial_condition=initial_condition.simple,
            parameters=[
                model_parameters.alpha,
                model_parameters.beta,
                model_parameters.sigma,
                model_parameters.gamma1,
                model_parameters.gamma2,
                model_parameters.theta_plus,
                model_parameters.theta_minus,
            ],
        ),
        'vaccine': _ODESystem(
            system=vaccine.system,
            initial_condition=initial_condition.vaccine,
            parameters=[
                model_parameters.alpha,
                model_parameters.beta,
                model_parameters.sigma,
                model_parameters.gamma1,
                model_parameters.gamma2,
                model_parameters.theta_plus,
                model_parameters.theta_minus,
                model_parameters.unprotected_lr,
                model_parameters.protected_all_types_lr,
                model_parameters.immune_all_types_lr,
                model_parameters.unprotected_hr,
                model_parameters.protected_all_types_hr,
                model_parameters.immune_all_types_hr,
            ]
        )
    }
    variant_systems = {
        'variant_explicit': variant.variant_explicit_system,
        'variant_implicit': variant.variant_implicit_system,
    }
    for system_name, system in variant_systems.items():
        systems[system_name] = _ODESystem(
            system=system,
            initial_condition=initial_condition.variant,
            parameters=[
                model_parameters.alpha,
                model_parameters.beta_wild,
                model_parameters.beta_b117,
                model_parameters.beta_b1351,
                model_parameters.beta_p1,
                model_parameters.sigma,
                model_parameters.gamma1,
                model_parameters.gamma2,
                model_parameters.theta_plus,
                model_parameters.theta_minus,
                model_parameters.b117_prevalence,
                model_parameters.b1351_prevalence,
                model_parameters.p1_prevalence,
                model_parameters.probability_cross_immune,
                model_parameters.unprotected_lr,
                model_parameters.protected_wild_type_lr,
                model_parameters.protected_all_types_lr,
                model_parameters.immune_wild_type_lr,
                model_parameters.immune_all_types_lr,
                model_parameters.unprotected_hr,
                model_parameters.protected_wild_type_hr,
                model_parameters.protected_all_types_hr,
                model_parameters.immune_wild_type_hr,
                model_parameters.immune_all_types_hr,
            ]
        )

    system, initial_condition, parameters = systems[system_type]
    parameters = pd.concat(parameters, axis=1)

    forecasts = []
    initial_conditions = tqdm.tqdm(initial_condition.iterrows(), total=len(initial_condition), disable=not progress_bar)
    for location_id, init_cond in initial_conditions:
        # Index columns to ensure sort order
        init_cond = init_cond.values

        loc_parameters = parameters.loc[location_id].sort_index()
        loc_date = loc_parameters.reset_index().date
        loc_times = np.array((loc_date - loc_date.min()).dt.days)

        p = loc_parameters.values.T  # Each row is a param, each column a day

        solution = math.solve_ode(
            system=system,
            t=loc_times,
            init_cond=init_cond,
            params=p,
            dt=dt,
        )

        result = pd.DataFrame(
            data=solution.T,
            columns=initial_condition.columns.tolist()
        )
        result['date'] = loc_date
        result['location_id'] = location_id
        forecasts.append(result.set_index(['location_id', 'date']))
    forecasts = pd.concat(forecasts).sort_index()
    return forecasts

from typing import Dict, Tuple, TYPE_CHECKING
import itertools

import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    math,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    ModelParameters,
    PostprocessingParameters,
    RatioData,
    HospitalCorrectionFactors,
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

def redistribute_past_compartments(compartments: pd.DataFrame,
                                   population: pd.DataFrame,
                                   model_parameters: ModelParameters):
    pop_weights = _get_pop_weights(population)
    unprotected_weights = _get_unprotected_vaccine_weights(model_parameters)

    redistributed_compartments = []
    for group in ['lr', 'hr']:
        # Need to broadcast pop weights.
        pop_weight = pop_weights[group].reindex(compartments.index, level='location_id')
        # vaccinations just needs a subset.
        unprotected_weight = unprotected_weights[group].loc[compartments.index]

        group_compartments = compartments.mul(pop_weight, axis=0)
        other_vacc_columns = [c for c in group_compartments if '_u' in c]
        protected_compartments = [c.replace('_u', '_p') for c in other_vacc_columns]

        group_compartments.loc[:, protected_compartments] = (group_compartments
                                                             .loc[:, other_vacc_columns]
                                                             .mul(1 - unprotected_weight, axis=0)
                                                             .rename(columns=dict(zip(other_vacc_columns,
                                                                                      protected_compartments))))
        group_compartments.loc[:, other_vacc_columns] = (group_compartments
                                                         .loc[:, other_vacc_columns]
                                                         .mul(unprotected_weight, axis=0))
        # sort the columns
        group_compartments = group_compartments.loc[:, vaccine.COMPARTMENTS]
        group_compartments.columns = [f'{c}_{group}' for c in group_compartments]
        redistributed_compartments.append(group_compartments)
    redistributed_compartments = pd.concat(redistributed_compartments, axis=1)

    return redistributed_compartments


def _get_pop_weights(population: pd.DataFrame) -> Dict[str, pd.Series]:
    total_pop = population.groupby('location_id')['population'].sum()
    low_risk_pop = population[population['age_group_years_start'] < 65].groupby('location_id')['population'].sum()
    high_risk_pop = total_pop - low_risk_pop
    pop_weights = {
        'lr': low_risk_pop / total_pop,
        'hr': high_risk_pop / total_pop,
    }
    return pop_weights


def _get_unprotected_vaccine_weights(model_parameters: ModelParameters) -> Dict[str, pd.Series]:
    mp_dict = model_parameters.to_dict()
    non_immune_cols = [
        'unprotected',
        'protected_wild_type',
        'protected_all_types',
    ]
    unprotected_vaccine_weights = {}
    for risk_group in ['lr', 'hr']:
        non_immune_vaccines = sum([mp_dict[f'{c}_{risk_group}'] for c in non_immune_cols])
        unprotected_weight = (mp_dict[f'unprotected_{risk_group}'] / non_immune_vaccines).fillna(0)
        unprotected_vaccine_weights[risk_group] = unprotected_weight
    return unprotected_vaccine_weights


#######################################
# Construct postprocessing parameters #
#######################################


def build_postprocessing_parameters(indices: Indices,
                                    past_compartments: pd.DataFrame,
                                    past_infections: pd.Series,
                                    past_deaths: pd.Series,
                                    betas: pd.DataFrame,
                                    ratio_data: RatioData,
                                    model_parameters: ModelParameters,
                                    correction_factors: HospitalCorrectionFactors,
                                    hospital_parameters: 'HospitalParameters',
                                    scenario_spec: 'ScenarioSpecification') -> PostprocessingParameters:
    ratio_data = correct_ratio_data(indices, ratio_data, model_parameters, scenario_spec.variant_ifr_scale)

    correction_factors = forecast_correction_factors(
        indices,
        correction_factors,
        hospital_parameters,
    )

    return PostprocessingParameters(
        past_beta=betas['beta'],
        past_compartments=past_compartments,
        past_infections=past_infections,
        past_deaths=past_deaths,
        **ratio_data.to_dict(),
        **correction_factors.to_dict()
    )


def correct_ratio_data(indices: Indices,
                       ratio_data: RatioData,
                       model_params: ModelParameters,
                       ifr_scale: float) -> RatioData:
    variant_prevalence = model_params.p_all_variant
    ifr_scalar = ifr_scale * variant_prevalence + (1 - variant_prevalence)

    ratio_data.ifr = ifr_scalar * _expand_rate(ratio_data.ifr, indices.full)
    ratio_data.ifr_lr = ifr_scalar * _expand_rate(ratio_data.ifr_lr, indices.full)
    ratio_data.ifr_hr = ifr_scalar * _expand_rate(ratio_data.ifr_hr, indices.full)

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



def get_component_groups(model_parameters: ModelParameters):
    new_e = infection_data.loc[:, 'infections'].groupby('location_id').fillna(0)

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


###########
# Run ODE #
###########

def run_ode_model(initial_conditions: pd.DataFrame,
                  model_parameters: ModelParameters,
                  progress_bar: bool) -> pd.DataFrame:
    system = vaccine.system
    mp_dict = model_parameters.to_dict()

    parameters = pd.concat([mp_dict[p] for p in vaccine.PARAMETERS]
                           + [model_parameters.unprotected_lr,
                              model_parameters.protected_all_types_lr,
                              model_parameters.immune_all_types_lr,
                              model_parameters.unprotected_hr,
                              model_parameters.protected_all_types_hr,
                              model_parameters.immune_all_types_hr], axis=1)

    forecasts = []
    initial_conditions_iter = tqdm.tqdm(initial_conditions.iterrows(),
                                        total=len(initial_conditions),
                                        disable=not progress_bar)
    for location_id, initial_condition in initial_conditions_iter:
        loc_parameters = parameters.loc[location_id].sort_index()
        loc_date = loc_parameters.reset_index().date
        loc_times = np.array((loc_date - loc_date.min()).dt.days)

        ic = initial_condition.values
        p = loc_parameters.values.T  # Each row is a param, each column a day

        solution = math.solve_ode(
            system=system,
            t=loc_times,
            init_cond=ic,
            params=p
        )

        result = pd.DataFrame(
            data=solution.T,
            columns=initial_conditions.columns.tolist()
        )
        result['date'] = loc_date
        result['location_id'] = location_id
        forecasts.append(result.set_index(['location_id', 'date']))
    forecasts = pd.concat(forecasts).sort_index()
    return forecasts

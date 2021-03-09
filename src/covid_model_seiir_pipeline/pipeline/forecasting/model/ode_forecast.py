from typing import Dict, Tuple, TYPE_CHECKING

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
from covid_model_seiir_pipeline.pipeline.forecasting.model import (
    ode_system,
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
                           scenario_spec: 'ScenarioSpecification',
                           draw_id: int) -> ModelParameters:
    # These are all the same by draw.  Just broadcasting them over a new index.
    alpha = pd.Series(ode_parameters.alpha.mean(), index=indices.full, name='alpha')
    sigma = pd.Series(ode_parameters.sigma.mean(), index=indices.full, name='sigma')
    gamma1 = pd.Series(ode_parameters.gamma1.mean(), index=indices.full, name='gamma1')
    gamma2 = pd.Series(ode_parameters.gamma2.mean(), index=indices.full, name='gamma2')

    np.random.seed(draw_id)
    variant_beta_scale = np.random.uniform(*scenario_spec.variant_beta_scale)
    probability_cross_immune = pd.Series(
        np.random.uniform(*scenario_spec.probability_cross_immune),
        index=indices.full, name='probability_cross_immune'
    )

    beta, beta_wild, beta_variant, p_wild, p_variant, p_all_variant = get_betas_and_prevalences(
        indices,
        beta_regression,
        covariates,
        coefficients,
        beta_scales,
        variant_beta_scale,
    )

    thetas = thetas.reindex(indices.full, level='location_id')

    if ((1 < thetas) | thetas < -1).any():
        raise ValueError('Theta must be between -1 and 1.')
    if (sigma - thetas >= 1).any():
        raise ValueError('Sigma - theta must be smaller than 1')

    theta_plus = np.maximum(thetas, 0).rename('theta_plus')
    theta_minus = -np.minimum(thetas, 0).rename('theta_minus')

    vaccine_data = vaccine_data.reindex(indices.full, fill_value=0)
    adjusted_vaccinations = math.adjust_vaccinations(vaccine_data)

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

def redistribute_past_compartments(infections: pd.Series,
                                   compartments: pd.DataFrame,
                                   population: pd.DataFrame,
                                   model_parameters: ModelParameters):
    pop_weights = _get_pop_weights(population)
    variant_prevalence = model_parameters.p_variant.loc[compartments.index]
    p_ci = model_parameters.probability_cross_immune.loc[compartments.index]

    redistributed_compartments = []
    for group in ['lr', 'hr']:
        # Need to broadcast pop weights.
        pop_weight = pop_weights[group].reindex(compartments.index, level='location_id')

        group_compartments = compartments.mul(pop_weight, axis=0)
        group_compartments = group_compartments.reindex(ode_system.COMPARTMENTS, axis='columns', fill_value=0.0)
        s_start = group_compartments.groupby('location_id')['S'].max()
        group_compartments_diff = group_compartments.groupby('location_id').diff()

        # R needs to to be redistributed in diff space since it's a sink only.
        group_compartments_diff = redistribute('R', group_compartments_diff, variant_prevalence)

        # Tracking compartments
        infecs = infections.reindex(group_compartments_diff.index)
        s_wild = group_compartments[['S', 'S_u', 'S_p', 'S_pa']].sum(axis=1)
        s_wild_p = group_compartments[['S_p', 'S_pa']].sum(axis=1)
        group_compartments_diff['NewE_wild'] = infecs * pop_weight * (1 - variant_prevalence)
        group_compartments_diff['NewE_p_wild'] = infecs * pop_weight * (1 - variant_prevalence) * s_wild_p / s_wild

        s_variant = s_wild + group_compartments[['S_variant', 'S_variant_u', 'S_variant_pa', 'S_m']].sum(axis=1)
        s_variant_p = group_compartments[['S_pa', 'S_variant_pa', 'S_m']].sum(axis=1)
        group_compartments_diff['NewE_variant'] = infecs * pop_weight * variant_prevalence
        group_compartments_diff['NewE_p_variant'] = infecs * pop_weight * variant_prevalence * s_variant_p / s_variant

        group_compartments = group_compartments_diff.groupby('location_id').cumsum().fillna(0)
        # Because these compartments have inflows and outflows and people spend a short time in them,
        # the best approximation is a redistribution in cumulative space.
        for compartment in ['E', 'I1', 'I2']:
            group_compartments = redistribute(compartment, group_compartments, variant_prevalence)

        # Who's in R vs. S_variant depends roughly on the probability of cross immunity.
        # This is a bad approximation if variant prevalence is high and there have been a significant
        # of infections.
        group_compartments['S_variant'] = (
            (group_compartments['R'] + group_compartments['R_variant']) * (1 - p_ci) - group_compartments['R_variant']
        )
        group_compartments['R'] = group_compartments['R'] * p_ci

        group_compartments['S_variant_u'] = (
            group_compartments[['R_u', 'R_p', 'R_variant_u']].sum(axis=1) * (1 - p_ci)
            - group_compartments['R_variant_u']
        )
        group_compartments['R_u'] = group_compartments['R_u'] * p_ci
        group_compartments['R_p'] = group_compartments['R_p'] * p_ci

        group_compartments['S_variant_pa'] = (
                group_compartments[['R_pa', 'R_variant_pa']].sum(axis=1) * (1 - p_ci)
                - group_compartments['R_variant_pa']
        )
        group_compartments_diff['R_pa'] = group_compartments_diff['R_pa'] * p_ci

        group_compartments['S'] += s_start.reindex(group_compartments.index, level='location_id')

        group_compartments['V_u'] = group_compartments[[c for c in group_compartments if '_u' in c]].sum(axis=1)
        group_compartments['V_p'] = group_compartments[[c for c in group_compartments if '_p' in c and '_pa' not in c]].sum(axis=1)
        group_compartments['V_pa'] = group_compartments[[c for c in group_compartments if '_pa' in c]].sum(axis=1)
        group_compartments['V_m'] = group_compartments['S_m']
        group_compartments['V_ma'] = group_compartments['R_m']

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


def redistribute(compartment: str, components: pd.DataFrame, variant_prevalence: pd.Series) -> pd.DataFrame:
    components[f'{compartment}_variant'] = components[compartment] * variant_prevalence
    components[f'{compartment}'] = components[compartment] * (1 - variant_prevalence)

    components[f'{compartment}_variant_u'] = (
        (components[f'{compartment}_u'] + components[f'{compartment}_p']) * variant_prevalence
    )
    components[f'{compartment}_u'] = components[f'{compartment}_u'] * (1 - variant_prevalence)
    components[f'{compartment}_p'] = components[f'{compartment}_p'] * (1 - variant_prevalence)

    components[f'{compartment}_variant_pa'] = components[f'{compartment}_pa'] * variant_prevalence
    components[f'{compartment}_pa'] = components[f'{compartment}_pa'] * (1 - variant_prevalence)
    return components


def adjust_beta(model_parameters: ModelParameters,
                initial_condition: pd.DataFrame,
                new_e: pd.Series) -> ModelParameters:

    s_wild = initial_condition[
        [c for c in initial_condition if c[0] == 'S' and 'variant' not in c and 'm' not in c]
    ].sum(axis=1)
    s_variant = initial_condition[
        [c for c in initial_condition if c[0] == 'S']
    ].sum(axis=1)
    i_wild = initial_condition[
        [c for c in initial_condition if c[0] == 'I' and 'variant' not in c]
    ].sum(axis=1)
    i_variant = initial_condition[
        [c for c in initial_condition if c[0] == 'I' and 'variant' in c]
    ].sum(axis=1)
    total_pop = initial_condition[
        [c for c in initial_condition if c[:-3] not in ode_system.TRACKING_COMPARTMENTS]
    ].sum(axis=1)

    variant_prevalence = model_parameters.p_variant.loc[new_e.index]
    alpha = model_parameters.alpha.loc[new_e.index]

    beta_wild = new_e * (1 - variant_prevalence) / (s_wild * i_wild**alpha / total_pop)
    beta_variant = new_e * variant_prevalence / (s_variant * i_variant**alpha / total_pop)

    wild_correction_factor = (
        beta_wild - model_parameters.beta_wild.loc[beta_wild.index]
    ).fillna(0).reset_index(level='date', drop=True)
    variant_correction_factor = (
        beta_variant - model_parameters.beta_variant.loc[beta_variant.index]
    ).fillna(0).reset_index(level='date', drop=True)

    idx = model_parameters.beta_wild.index
    model_parameters.beta_wild = (
        model_parameters.beta_wild + wild_correction_factor.reindex(idx, level='location_id')
    )
    model_parameters.beta_variant = (
            model_parameters.beta_variant + variant_correction_factor.reindex(idx, level='location_id')
    )

    return model_parameters


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
    p_start = variant_prevalence.loc[indices.initial_condition].reset_index(level='date', drop=True)
    variant_prevalence -= p_start.reindex(variant_prevalence.index, level='location_id')
    variant_prevalence[variant_prevalence < 0] = 0.0
    ifr_scalar = ifr_scale * variant_prevalence + (1 - variant_prevalence)
    ifr_scalar = ifr_scalar.groupby('location_id').shift(ratio_data.infection_to_death).fillna(0.)

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


###########
# Run ODE #
###########

def run_ode_model(initial_conditions: pd.DataFrame,
                  model_parameters: ModelParameters,
                  progress_bar: bool) -> pd.DataFrame:
    system = ode_system.variant_natural_system
    mp_dict = model_parameters.to_dict()

    parameters = pd.concat(
        [mp_dict[p] for p in ode_system.PARAMETERS]
        + [model_parameters.unprotected_lr,
           model_parameters.protected_wild_type_lr,
           model_parameters.protected_all_types_lr,
           model_parameters.immune_wild_type_lr,
           model_parameters.immune_all_types_lr,

           model_parameters.unprotected_hr,
           model_parameters.protected_wild_type_hr,
           model_parameters.protected_all_types_hr,
           model_parameters.immune_wild_type_hr,
           model_parameters.immune_all_types_hr],
        axis=1
    )

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

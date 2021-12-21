from typing import List, Tuple

import numpy as np
import pandas as pd


from covid_model_seiir_pipeline.lib import math
from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    SYSTEM_TYPE,
    RISK_GROUP_NAMES,
    VARIANT_NAMES,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    solver,
)
from covid_model_seiir_pipeline.pipeline.regression.model.hospital_corrections import (
    HospitalCorrectionFactors,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
)


##############################
# ODE parameter construction #
##############################

def build_beta_final(indices: Indices,
                     beta_regression: pd.DataFrame,
                     covariates: pd.DataFrame,
                     coefficients: pd.DataFrame,
                     beta_shift_parameters: pd.DataFrame,
                     log_beta_shift: Tuple[float, pd.Timestamp],
                     beta_scale: Tuple[float, pd.Timestamp]):
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
    log_beta_hat.loc[pd.IndexSlice[:, log_beta_shift[1]:]] += log_beta_shift[0]
    beta_hat = np.exp(log_beta_hat).loc[indices.future].rename('beta_hat').reset_index()

    beta = (beta_shift(beta_hat, beta_shift_parameters)
            .set_index(['location_id', 'date'])
            .beta_hat
            .rename('beta'))
    beta = beta_regression.reindex(indices.past).loc[:, 'beta'].append(beta).sort_index()
    beta.loc[pd.IndexSlice[:, beta_scale[1]:]] *= beta_scale[0]
    return beta, beta_hat.set_index(['location_id', 'date']).reindex(beta.index)


def build_model_parameters(indices: Indices,
                           beta: pd.Series,
                           past_compartments: pd.DataFrame,
                           prior_ratios: pd.DataFrame,
                           ode_parameters: pd.DataFrame,
                           rhos: pd.DataFrame,
                           vaccinations: pd.DataFrame,
                           etas: pd.DataFrame,
                           phis: pd.DataFrame) -> Parameters:
    ode_params = ode_parameters.reindex(indices.full).groupby('location_id').ffill().groupby('location_id').bfill()
    ode_params.loc[:, 'beta_all_infection'] = beta

    ode_params = ode_params.drop(columns=[c for c in ode_params if 'rho' in c])
    rhos = og_rhos.reindex(indices.full, fill_value=0.)
    rhos.columns = [f'rho_{c}_infection' for c in rhos.columns]
    rhos.loc[:, 'rho_none_infection'] = 0
    base_parameters = pd.concat([ode_params, rhos], axis=1)
    
    empirical_rhos = pd.concat([
        (past_compartments.filter(like=f'Infection_all_{v}_all').diff().sum(axis=1, min_count=1)
         / past_compartments.filter(like='Infection_all_all_all').diff().sum(axis=1, min_count=1)).rename(v)
        for v in VARIANT_NAMES[1:]
    ], axis=1)
    
    ratio_map = {
        'death': 'ifr',
        'admission': 'ihr',
        'case': 'idr',
    }

    prior_ratios = prior_ratios.loc[prior_ratios['round'] == 2]
    scalars = []
    infections = (past_compartments
                  .filter(like='Infection_none_all_unvaccinated')
                  .sum(axis=1, min_count=1)
                  .reindex(indices.full))
    infections = infections.groupby('location_id').diff().fillna(infections)

    for epi_measure, ratio_name in ratio_map.items():
        ode_params.loc[:, f'count_all_{epi_measure}'] = -1
        ode_params.loc[:, f'weight_all_{epi_measure}'] = -1
        # Same for all location-dates
        numerator = (past_compartments
                     .filter(like=f'{epi_measure.capitalize()}_none_all_unvaccinated')
                     .sum(axis=1, min_count=1)
                     .reindex(indices.full))
        numerator = numerator.groupby('location_id').diff().fillna(numerator)
        prior_ratio = prior_ratios.loc[:, ratio_name].groupby('location_id').last()
        prior_ratio = prior_ratios.loc[:, ratio_name].groupby('location_id').last()
        kappas = (ode_params
                  .loc[empirical_rhos.index, [f'kappa_{variant}_{epi_measure}' for variant in VARIANT_NAMES[1:]]]
                  .rename(columns=lambda x: x.split('_')[1]))
        ode_params.loc[:, f'rate_all_{epi_measure}'] = build_ratio(
            epi_measure,
            infections,
            numerator,
            prior_ratio,
            empirical_rhos,
            kappas,
        )

        for risk_group in RISK_GROUP_NAMES:
            scalars.append(
                (prior_ratios[f'{ratio_name}_{risk_group}'] / prior_ratios[ratio_name])
                .rename(f'{epi_measure}_{risk_group}')
                .reindex(indices.full)
                .groupby('location_id')
                .ffill()
                .groupby('location_id')
                .bfill()
            )
    scalars = pd.concat(scalars, axis=1)



    vaccinations = vaccinations.reindex(indices.full, fill_value=0.)
    etas = etas.sort_index().reindex(indices.full, fill_value=0.)

    return Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        age_scalars=scalars,
        etas=etas,
        phis=phis,
    )


def build_ratio(epi_measure: str,
                infections: pd.Series,
                shifted_numerator: pd.Series,
                prior_ratio: pd.Series,
                rhos: pd.DataFrame,
                kappas: pd.DataFrame):
    posterior_ratio = (shifted_numerator / infections).rename('value')
    posterior_ratio.loc[(posterior_ratio == 0) | ~np.isfinite(posterior_ratio)] = np.nan
    locs = posterior_ratio.reset_index().location_id.unique()    
    for location_id in locs:
        count = posterior_ratio.loc[location_id].notnull().sum()
        if not count:
            try:
                posterior_ratio.loc[location_id, :] = prior_ratio.loc[location_id]
            except KeyError:
                pass
    
    correction = 1 / (empirical_rhos * kappas).sum(axis=1, min_count=1)
    ancestral_ratio = (posterior_ratio * correction).rename('value')    

    pr_gb = ancestral_ratio.dropna().reset_index().groupby('location_id')
    date = pr_gb.date.last()
    final_ancestral_ratio = pr_gb.value.last()

    past_window = 180
    ancestral_infections = ((infections * rhos['ancestral'])
                            .replace(0, np.nan)
                            .rename('denom'))
    ancestral_numerator = ((ancestral_ratio * ancestral_infections)
                           .rename('num'))
    lr_ratio = pd.concat([ancestral_numerator, ancestral_infections], axis=1)
    lr_ratio = (lr_ratio
                .dropna()
                .groupby('location_id')
                .apply(lambda x: x.iloc[-past_window:].num.sum() / x.iloc[-past_window:].denom.sum()))

    trans_window = 30
    scale = (lr_ratio - final_ancestral_ratio) / trans_window
    t = pd.Series(np.tile(np.arange(trans_window + 1), len(scale)),
                  index=pd.MultiIndex.from_product((locs, np.arange(trans_window + 1)),
                                                   names=('location_id', 't')))
    rate_scaleup = (final_ancestral_ratio + scale * t).rename('value').reset_index(level='t')
    rate_scaleup['date'] = pd.to_timedelta(rate_scaleup['t'], unit='D') + date
    rate_scaleup = rate_scaleup.set_index('date', append=True).value
    ratio = (ancestral_ratio
             .drop(ancestral_ratio.index.intersection(rate_scaleup.index))
             .append(rate_scaleup)
             .sort_index()
             .reindex(ancestral_infections.index)
             .groupby('location_id')
             .ffill()
             .groupby('location_id')
             .bfill())

    return ratio


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


def forecast_correction_factors(indices: Indices,
                                correction_factors,
                                hospital_parameters):
    averaging_window = pd.Timedelta(days=hospital_parameters.correction_factor_average_window)
    application_window = pd.Timedelta(days=hospital_parameters.correction_factor_application_window)

    new_cfs = {}
    for cf_name, cf in correction_factors.to_dict('series').items():
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

def run_ode_forecast(initial_condition: pd.DataFrame,
                     ode_parameters: Parameters,
                     num_cores: int,
                     progress_bar: bool,
                     location_ids: List[int] = None):
    if location_ids is None:
        location_ids = initial_condition.reset_index().location_id.unique().tolist()
    full_compartments, chis = solver.run_ode_model(
        initial_condition,
        **ode_parameters.to_dict(),
        location_ids=location_ids,
        system_type=SYSTEM_TYPE.beta_and_rates,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    return full_compartments, chis

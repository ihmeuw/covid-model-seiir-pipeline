from typing import Tuple

from loguru import logger
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from covid_model_seiir_pipeline.lib import math
from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters
from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import (
    Durations,
)
from covid_model_seiir_pipeline.pipeline.fit.model.rates.mrbrt import (
    mrbrt,
)

ASSAYS = [
    'N-Abbott',  # IgG
    'S-Roche', 'N-Roche',  # Ig
    'S-Ortho Ig', 'S-Ortho IgG',  # Ig/IgG
    'S-DiaSorin',  # IgG
    'S-EuroImmun',  # IgG
    'S-Oxford',  # IgG
]
INCREASING = ['S-Ortho Ig', 'S-Roche']
SEROPREV_LB = 0.


def subset_seroprevalence(seroprevalence: pd.DataFrame,
                          epi_data: pd.DataFrame,
                          variant_prevalence: pd.Series,
                          population: pd.Series,
                          params: RatesParameters) -> pd.DataFrame:
    cumulative_deaths = epi_data['cumulative_deaths'].dropna()
    death_threshold = params.death_rate_threshold / 1e6
    death_dates = (cumulative_deaths.loc[(cumulative_deaths / population) < death_threshold]
                   .reset_index()
                   .groupby('location_id')['date'].max()
                   .rename('death_date'))
    variant_dates = (variant_prevalence.loc[variant_prevalence < params.variant_prevalence_threshold]
                     .reset_index()
                     .groupby('location_id')['date'].max()
                     .rename('variant_date'))
    invasion_dates = pd.concat([death_dates, variant_dates], axis=1)
    invasion_dates = (invasion_dates
                      .fillna(invasion_dates.max().max())
                      .min(axis=1)
                      .rename('invasion_date'))
    inclusion_date = ((invasion_dates + pd.Timedelta(days=params.inclusion_days))
                      .rename('inclusion_date')
                      .reset_index())
    seroprevalence = seroprevalence.merge(inclusion_date)

    seroprevalence = seroprevalence.loc[seroprevalence['date'] <= seroprevalence['inclusion_date']]
    del seroprevalence['inclusion_date']

    return seroprevalence


def apply_sensitivity_adjustment(sensitivity_data: pd.DataFrame,
                                 hospitalized_weights: pd.Series,
                                 seroprevalence: pd.DataFrame,
                                 daily_infections: pd.Series,
                                 population: pd.Series,
                                 durations: Durations) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_assays = sensitivity_data['assay'].unique().tolist()
    excluded_data_assays = [da for da in data_assays if da not in ASSAYS]
    if excluded_data_assays:
        logger.warning(f"Excluding the following assays found in sensitivity data: {excluded_data_assays}")
    if any([a not in data_assays for a in ASSAYS]):
        raise ValueError('Assay mis-labelled.')

    sensitivity_data = sensitivity_data.loc[sensitivity_data['assay'].isin(ASSAYS)]
    source_assays = sensitivity_data[['source', 'assay']].drop_duplicates().values.tolist()

    sensitivity = []
    for source_assay in source_assays:
        sensitivity.append(
            fit_hospital_weighted_sensitivity_decay(
                source_assay,
                sensitivity=sensitivity_data.set_index(['source', 'assay']).loc[tuple(source_assay)],
                hospitalized_weights=hospitalized_weights.copy(),
            )
        )
    sensitivity = pd.concat(sensitivity).set_index(['assay', 'location_id', 't']).sort_index()

    seroprevalence = seroprevalence.loc[seroprevalence['is_outlier'] == 0]
    assay_combinations = seroprevalence['assay_map'].unique().tolist()

    daily_infections = daily_infections.reset_index()
    daily_infections['date'] += pd.Timedelta(days=durations.exposure_to_seroconversion)
    daily_infections = daily_infections.set_index(['location_id', 'date']).loc[:, 'daily_infections']
    daily_infections /= population

    sensitivity_list = []
    seroprevalence_list = []
    for assay_combination in assay_combinations:
        logger.info(f'Adjusting for sensitvity decay: {assay_combination}')
        ac_sensitivity = (sensitivity
                          .loc[assay_combination.split(', ')]
                          .reset_index()
                          .groupby(['location_id', 't'])['sensitivity'].mean())
        ac_seroprevalence = (seroprevalence
                             .loc[seroprevalence['assay_map'] == assay_combination].copy())
        ac_seroprevalence = sensitivity_adjustment(
            daily_infections.copy(),
            ac_sensitivity.copy(),
            ac_seroprevalence.copy(),
        )

        ac_sensitivity = (ac_sensitivity
                          .loc[ac_seroprevalence['location_id'].unique().tolist()]
                          .reset_index())
        ac_sensitivity['assay'] = assay_combination
        sensitivity_list.append(ac_sensitivity)

        ac_seroprevalence['is_outlier'] = 0
        ac_seroprevalence['assay'] = assay_combination
        seroprevalence_list.append(ac_seroprevalence)
    sensitivity = pd.concat(sensitivity_list)
    seroprevalence = pd.concat(seroprevalence_list)

    return sensitivity, seroprevalence


def fit_hospital_weighted_sensitivity_decay(source_assay: Tuple[str, str],
                                            sensitivity: pd.DataFrame,
                                            hospitalized_weights: pd.Series) -> pd.DataFrame:
    source, assay = source_assay

    increasing = assay in INCREASING

    if source not in ['Peluso', 'Perez-Saez', 'Bond', 'Muecksch', 'Lumley']:
        raise ValueError(f'Unexpected sensitivity source: {source}')

    hosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Hospitalized']
    nonhosp_sensitivity = sensitivity.loc[sensitivity['hospitalization_status'] == 'Non-hospitalized']
    if source == 'Peluso':
        hosp_sensitivity = fit_sensitivity_decay_curvefit(hosp_sensitivity['t'].values,
                                                          hosp_sensitivity['sensitivity'].values,
                                                          increasing, )
        nonhosp_sensitivity = fit_sensitivity_decay_curvefit(nonhosp_sensitivity['t'].values,
                                                             nonhosp_sensitivity['sensitivity'].values,
                                                             increasing, )
    else:
        hosp_sensitivity = fit_sensitivity_decay_mrbrt(hosp_sensitivity.loc[:, ['t', 'sensitivity']],
                                                       increasing, )
        nonhosp_sensitivity = fit_sensitivity_decay_mrbrt(hosp_sensitivity.loc[:, ['t', 'sensitivity']],
                                                          increasing, )
    sensitivity = (hosp_sensitivity
                   .rename(columns={'sensitivity': 'hosp_sensitivity'})
                   .merge(nonhosp_sensitivity
                          .rename(columns={'sensitivity': 'nonhosp_sensitivity'})))
    sensitivity['key'] = 0
    hospitalized_weights = hospitalized_weights.rename('hospitalized_weights')
    hospitalized_weights = hospitalized_weights.reset_index()
    hospitalized_weights['key'] = 0
    sensitivity = sensitivity.merge(hospitalized_weights, on='key', how='outer')

    sensitivity['hosp_sensitivity'] = math.scale_to_bounds(sensitivity['hosp_sensitivity'],
                                                           SEROPREV_LB, 1.)
    sensitivity['nonhosp_sensitivity'] = math.scale_to_bounds(sensitivity['nonhosp_sensitivity'],
                                                              SEROPREV_LB, 1.)
    sensitivity['sensitivity'] = (
            (sensitivity['hosp_sensitivity'] * sensitivity['hospitalized_weights'])
            + (sensitivity['nonhosp_sensitivity'] * (1 - sensitivity['hospitalized_weights']))
    )
    sensitivity = sensitivity.reset_index()

    sensitivity['assay'] = assay

    return sensitivity.loc[:, ['location_id', 'assay', 't', 'sensitivity', 'hosp_sensitivity', 'nonhosp_sensitivity']]


def fit_sensitivity_decay_curvefit(t: np.array, sensitivity: np.array, increasing: bool,
                                   t_N: int = 1080) -> pd.DataFrame:
    def sigmoid(x, x0, k):
        y = 1 / (1 + np.exp(-k * (x - x0)))
        return y

    if increasing:
        bounds = ([-np.inf, 1e-4], [np.inf, 0.5])
    else:
        bounds = ([-np.inf, -0.5], [np.inf, -1e-4])
    popt, pcov = curve_fit(sigmoid,
                           t, sensitivity,
                           method='dogbox',
                           bounds=bounds, max_nfev=1000)

    t_pred = np.arange(0, t_N + 1)
    sensitivity_pred = sigmoid(t_pred, *popt)

    return pd.DataFrame({'t': t_pred, 'sensitivity': sensitivity_pred})


def fit_sensitivity_decay_mrbrt(sensitivity_data: pd.DataFrame, increasing: bool,
                                t_N: int = 1080) -> pd.DataFrame:
    sensitivity_data = sensitivity_data.loc[:, ['t', 'sensitivity', ]]
    sensitivity_data['sensitivity'] = math.logit(sensitivity_data['sensitivity'])
    sensitivity_data['intercept'] = 1
    sensitivity_data['se'] = 1
    sensitivity_data['location_id'] = 1

    if increasing:
        mono_dir = 'increasing'
    else:
        mono_dir = 'decreasing'

    n_k = min(max(len(sensitivity_data) - 3, 2), 10, )
    k = np.hstack([[0, 0.1], np.linspace(0.1, 1, n_k)[1:]])
    max_t = sensitivity_data['t'].max()

    mr_model = mrbrt.run_mr_model(
        model_data=sensitivity_data,
        dep_var='sensitivity', dep_var_se='se',
        fe_vars=['intercept', 't'], re_vars=[],
        group_var='location_id',
        prior_dict={'intercept': {},
                    't': {'use_spline': True,
                          'spline_knots_type': 'domain',
                          'spline_knots': np.linspace(0, 1, n_k),
                          'spline_degree': 1,
                          'prior_spline_monotonicity': mono_dir,
                          'prior_spline_monotonicity_domain': (60 / max_t, 1),
                          }, }
    )
    t_pred = np.arange(t_N + 1)
    sensitivity_pred, _ = mrbrt.predict(
        pred_data=pd.DataFrame({'intercept': 1,
                                't': t_pred,
                                'location_id': 1,
                                'date': t_pred, }),
        hierarchy=None,
        mr_model=mr_model,
        pred_replace_dict={},
        pred_exclude_vars=[],
        dep_var='sensitivity', dep_var_se='se',
        fe_vars=['t'], re_vars=[],
        group_var='location_id',
        sensitivity=True,
    )

    return pd.DataFrame({'t': t_pred, 'sensitivity': math.expit(sensitivity_pred['sensitivity'])})


def sensitivity_adjustment(daily_infections: pd.Series,
                           sensitivity: pd.DataFrame,
                           seroprevalence: pd.DataFrame) -> pd.DataFrame:
    location_ids = seroprevalence['location_id'].unique().tolist()
    location_ids = [location_id for location_id in location_ids if
                    location_id in daily_infections.reset_index()['location_id'].to_list()]

    out = []
    for location_id in location_ids:
        loc_seroprevalence = location_sensitivity_adjustment(
            location_id=location_id,
            daily_infections=daily_infections,
            sensitivity=sensitivity,
            seroprevalence=seroprevalence,
        )
        out.append(loc_seroprevalence)
    seroprevalence = pd.concat(out).reset_index(drop=True)

    return seroprevalence


def location_sensitivity_adjustment(location_id: int,
                                    daily_infections: pd.Series,
                                    sensitivity: pd.DataFrame,
                                    seroprevalence: pd.DataFrame) -> pd.DataFrame:
    daily_infections = daily_infections.loc[location_id].reset_index()
    sensitivity = sensitivity.loc[location_id]
    seroprevalence = (seroprevalence
                      .loc[seroprevalence['location_id'] == location_id,
                           ['data_id', 'date', 'manufacturer_correction', 'seroprevalence']]
                      .reset_index(drop=True))
    adj_seroprevalence = []
    for i, (sero_data_id, sero_date, sero_corr, sero_value) in enumerate(zip(seroprevalence['data_id'],
                                                                             seroprevalence['date'],
                                                                             seroprevalence['manufacturer_correction'],
                                                                             seroprevalence['seroprevalence'], )):
        seroreversion_factor = calculate_seroreversion_factor(daily_infections.copy(), sensitivity.copy(),
                                                              sero_date, sero_corr, )
        adj_seroprevalence.append(pd.DataFrame({
            'data_id': sero_data_id,
            'date': sero_date,
            'seroprevalence': 1 - (1 - sero_value) * seroreversion_factor
        }, index=[i]))
    adj_seroprevalence = pd.concat(adj_seroprevalence)
    adj_seroprevalence['location_id'] = location_id

    return adj_seroprevalence


def calculate_seroreversion_factor(daily_infections: pd.DataFrame,
                                   sensitivity: pd.DataFrame,
                                   sero_date: pd.Timestamp,
                                   sero_corr: bool) -> float:
    daily_infections['t'] = (sero_date - daily_infections['date']).dt.days
    daily_infections = daily_infections.loc[daily_infections['t'] >= 0]
    if sero_corr not in [0, 1]:
        raise ValueError('`manufacturer_correction` should be 0 or 1.')
    if sero_corr == 1:
        # study adjusted for sensitivity, set baseline to 1
        sensitivity /= sensitivity.max()
    daily_infections = daily_infections.merge(sensitivity.reset_index(), how='left')
    if daily_infections['sensitivity'].isnull().any():
        raise ValueError(f"Unmatched sero/sens points: {daily_infections.loc[daily_infections['sensitivity'].isnull()]}")
    daily_infections['daily_infections'] *= min(1, 1 / daily_infections['daily_infections'].sum())
    seroreversion_factor = (
        (1 - daily_infections['daily_infections'].sum())
        / (1 - (daily_infections['daily_infections'] * daily_infections['sensitivity']).sum())
    )
    seroreversion_factor = min(1, seroreversion_factor)

    return seroreversion_factor

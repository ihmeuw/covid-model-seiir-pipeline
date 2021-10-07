import itertools
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
    VARIANT_NAMES,
)


def prepare_etas_and_vaccinations(past_infections: pd.Series,
                                  vaccinations: pd.DataFrame, 
                                  waning_params: Tuple[int, float, int, float]):    
    inf_start_date = past_infections.reset_index().groupby('location_id').date.min()
    vac_start_date = vaccinations.reset_index().groupby('location_id').date.min() - pd.Timedelta(days=1)
    prepend_index = pd.concat([inf_start_date, vac_start_date]).reset_index().set_index(['location_id', 'date']).index
    prepend = (pd.DataFrame(0., columns=vaccinations.columns, index=prepend_index)
               .sort_index()
               .groupby('location_id')
               .apply(lambda x: x.reset_index('location_id', drop=True).asfreq('D'))
               .fillna(0.))
    vaccinations = pd.concat([prepend, vaccinations]).sort_index()

    total_vaccinations, efficacy = get_total_vaccinations_and_efficacy(vaccinations)
    efficacy = remap_efficacy(efficacy)
    waning = build_waning_dist(efficacy, waning_params)
    etas = compute_eta(total_vaccinations, efficacy, waning)

    etas, waning, total_vaccinations = map(pivot_risk_group, [etas, waning, total_vaccinations])
    
    etas_immune = etas[[c for c in etas.columns if 'immune' in c]]
    etas_immune.columns = ['_'.join([x for x in c.split('_') if x != 'immune']) for c in etas_immune.columns]
    
    etas_protected = etas[[c for c in etas.columns if 'protected' in c]]
    etas_protected.columns = ['_'.join([x for x in c.split('_') if x != 'protected']) for c in etas_protected.columns]
    
    return etas_immune, etas_protected, total_vaccinations


def prepare_phis(past_infections: pd.Series, 
                 covariates: pd.DataFrame,
                 waning_matrix: pd.DataFrame,
                 waning_params: Tuple[int, float, int, float]):
    min_date = past_infections.reset_index().date.min()
    max_date = covariates.reset_index().date.max()
    
    dates = pd.Series(pd.date_range(min_date, max_date))
    times = (dates - min_date).dt.days    
    max_t = times.max() + 1
    
    phi_columns = [f'phi_{v_from}_{v_to}' 
                   for v_from, v_to in itertools.product(waning_matrix, waning_matrix)]
    phi = pd.DataFrame(0., 
                       index=times,
                       columns=phi_columns)
    
    w = make_raw_waning_dist(max_t, *waning_params)
    
    for v_from in waning_matrix.index:
        for v_to in waning_matrix.columns:
            phi.loc[:, f'phi_{v_from}_{v_to}'] = waning_matrix.at[v_from, v_to] * w
    
    return phi


def get_total_vaccinations_and_efficacy(vaccinations: pd.DataFrame):
    """Create total vaccinations and average efficacy of delivered doses by variant."""
    ordered_efficacy_cols = ['omega_immune', 'escape_immune', 'non_escape_immune',
                             'omega_protected', 'escape_protected', 'non_escape_protected']
    v, e = [], []
    for risk_group in RISK_GROUP_NAMES:
        df = vaccinations.filter(like=risk_group).reset_index()
        df['risk_group'] = risk_group
        df = df.set_index(['location_id', 'risk_group', 'date'])
        total_vaccinations = df.sum(axis=1).rename(f'vaccinations')
        eps = []
        for i, eps_type in enumerate(ordered_efficacy_cols):
            cols = [f'vaccinations_{efficacy}_{risk_group}' for efficacy in ordered_efficacy_cols[:i + 1]]
            eps.append((df[cols].sum(axis=1) / total_vaccinations).fillna(0).rename(f'{eps_type}'))
        eps = pd.concat(eps, axis=1)
        v.append(total_vaccinations)
        e.append(eps)

    return pd.concat(v), pd.concat(e)


def remap_efficacy(efficacy: pd.DataFrame) -> pd.DataFrame:
    """Map old efficacy groups onto variants."""
    variant_mapping = {
        'ancestral': 'non_escape',
        'alpha': 'non_escape',
        'beta': 'escape',
        'gamma': 'escape',
        'delta': 'escape',
        'other': 'escape',
        'omega': 'omega',
    }
    assert set(variant_mapping) | {'none'} == set(VARIANT_NAMES)

    final_cols = []
    for variant, variant_group in variant_mapping.items():
        for status in ['protected', 'immune']:
            new_col = f'{variant}_{status}'
            efficacy[new_col] = efficacy[f'{variant_group}_{status}']
            final_cols.append(new_col)
    efficacy = efficacy.loc[:, final_cols]
    efficacy['none_protected'] = 0.
    efficacy['none_immune'] = 0.
    return efficacy


def build_waning_dist(efficacy: pd.DataFrame, waning_params: Tuple[int, float, int, float]):
    efficacy = convert_date_index_to_time_index(efficacy)
    waning = efficacy.copy()
    max_t = waning.reset_index().t.max() + 1
    for column in waning.columns:
        w = make_raw_waning_dist(max_t, *waning_params)
        w = pd.Series(w, index=pd.Index(np.arange(max_t), name='t')).reindex(waning.index, level='t')
        waning.loc[:, column] = w
    return waning


def make_raw_waning_dist(max_t, l1, t1, l2, t2):
    w = np.zeros(max_t)
    w[:t1] = 1 + (l1 - 1) / (t1 - 0) * np.arange(t1)
    w[t1:t2] = l1 + (l2 - l1) / (t2 - t1) * np.arange(t2 - t1)
    w[t2:] = l2
    return w


def compute_eta(total_vaccinations, efficacy, waning):
    ve = efficacy.mul(total_vaccinations, axis=0)
    eta = ve.copy()
    for location_id in tqdm.tqdm(ve.reset_index().location_id.unique()):
        for risk_group in ['hr', 'lr']:
            idx = (location_id, risk_group)
            eta.loc[idx] = _compute_eta(
                total_vaccinations.loc[idx].values,
                ve.loc[idx].values,
                waning.loc[idx].values,
            )
    return eta.groupby(['location_id', 'risk_group']).bfill().fillna(0.)


@numba.njit
def _compute_eta(v, ve, w):
    eta = np.zeros(ve.shape)
    for t in range(1, v.size + 1):
        eta[t - 1] = (ve[:t][::-1] * w[:t]).sum(axis=0) / v[:t].sum()
    return eta


def convert_date_index_to_time_index(data):
    data = data.sort_index()
    original_index = data.index
    idx_cols = original_index.names
    new_idx_cols = [c if c != 'date' else 't' for c in idx_cols]

    is_series = isinstance(data, pd.Series)

    data = data.reset_index()
    data['t'] = (data.date - data.date.min()).dt.days

    data = data.set_index(new_idx_cols).drop(columns='date')
    if is_series:
        data = data.iloc[:, 0]
    return data


def pivot_risk_group(data):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    new_levels = [n for n in data.index.names if n != 'risk_group'] + ['risk_group']
    data = data.reorder_levels(new_levels).sort_index().unstack()
    data.columns = ['_'.join(levels) for levels in data.columns]
    return data

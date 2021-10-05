import numba
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
    VARIANT_NAMES,
)


def prepare_etas_and_vaccinations(vaccinations: pd.DataFrame):
    total_vaccinations, efficacy = get_total_vaccinations_and_efficacy(vaccinations)
    waning_dist = build_waning_dist(efficacy)
    etas = compute_eta(total_vaccinations, efficacy, waning_dist)

    # TODO: pivot wide

    return etas, vaccinations


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
            cols = [f'vaccinations_{efficacy}' for efficacy in ordered_efficacy_cols[:i + 1]]
            eps.append((df[cols].sum(axis=1) / vaccinations).fillna(0).rename(f'{eps_type}'))
        eps = pd.concat(eps, axis=1)
        v.append(total_vaccinations)
        e.append(eps)

    return pd.concat(v, axis=1), pd.concat(e, axis=1)


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
    assert set(variant_mapping).add('none') == set(VARIANT_NAMES)

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


def build_waning_dist(efficacy: pd.DataFrame):
    fast = (0.5, 180, 0.1, 720)
    # moderate = (0.7, 180, 0.5, 720)
    # slow = (1.0, 180, 0.8, 720)
    waning_parameters = {c: fast for c in efficacy.columns}

    waning = efficacy.copy()
    max_t = waning.reset_index().t.max() + 1
    for parameter_name, params in waning_parameters.items():
        w = np.zeros(max_t)
        l1, t1, l2, t2 = params
        w[:t1] = 1 + (l1 - 1) / (t1 - 0) * np.arange(t1)
        w[t1:t2] = l1 + (l2 - l1) / (t2 - t1) * np.arange(t2 - t1)
        w[t2:] = l2
        w = pd.Series(w, index=pd.Index(np.arange(max_t), name='t')).reindex(waning.index, level='t')
        waning.loc[:, parameter_name] = w
    return waning


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
    return data, original_index


def compute_eta(total_vaccinations, efficacy, waning):
    ve = efficacy.mul(total_vaccinations, axis=0)
    eta = ve.copy()
    for location_id in ve.reset_index().location_id.unique():
        for risk_group in ['hr', 'lr']:
            idx = (location_id, risk_group)
            eta.loc[idx] = _compute_eta(
                total_vaccinations.loc[idx].values,
                ve.loc[idx].values,
                waning.loc[idx].values,
            )
    return eta


@numba.njit
def _compute_eta(v, ve, w):
    eta = np.zeros(ve.shape)
    for t in range(1, v.size + 1):
        eta[t - 1] = (ve[:t][::-1] * w[:t]).sum(axis=0) / v[:t].sum()
    return eta

"""ETL for vaccine coverage covariates and scenarios."""
import functools
import itertools
import multiprocessing
from pathlib import Path
from typing import Tuple

import numba
import numpy as np
import pandas as pd
import tqdm

from covid_shared import paths as paths

from covid_input_seir_covariates.utilities import CovariateGroup


COVARIATE_NAMES = (
    'vaccine_coverage',
)

DEFAULT_OUTPUT_ROOT = paths.VACCINE_COVERAGE_OUTPUT_ROOT


RISK_GROUP_NAMES = ['lr', 'hr']
WANING_PARAMS = (0.5, 180, 0.1, 720)


def get_covariates(covariates_root: Path) -> CovariateGroup:
    """Gathers and formats all testing covariates and scenarios."""
    # No scenarios here since vaccine coverage is not a beta covariate.
    scenarios = {}
    hesitancy_paths = [
        covariates_root / 'time_series_vaccine_hesitancy.csv',
        covariates_root / 'time_point_vaccine_hesitancy.csv',
    ]

    scenario_paths = {
        'reference': covariates_root / 'slow_scenario_vaccine_coverage.csv',
    }
    info = {}

    for scenario, scenario_path in scenario_paths.items():
        vaccinations, etas, summaries = load_vaccine_data(scenario_path, *hesitancy_paths)
        info[f'vaccinations_{scenario}'] = vaccinations
        info[f'etas_{scenario}'] = etas
        info[f'vaccinations_{scenario}_summaries'] = summaries

    info['vaccine_efficacy'] = pd.read_csv(covariates_root / 'vaccine_efficacy_table.csv')

    return {'vaccine_coverage': (scenarios, info)}


def load_vaccine_data(data_path: Path, series_path: Path, point_path: Path) -> Tuple[pd.DataFrame,
                                                                                     pd.DataFrame,
                                                                                     pd.DataFrame]:
    """Loads, validates, and cleans vaccine data."""
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['location_id', 'date']).sort_index()

    high_risk, low_risk = [extract_vaccine_categories(data, population_group)
                           for population_group in RISK_GROUP_NAMES]
    vaccinations = pd.concat([high_risk, low_risk], axis=1)    
    boosters = vaccinations.copy()
    boosters.loc[:, :] = 0.
    vaccinations['course'] = 1
    boosters['course'] = 2
    vaccinations = pd.concat([vaccinations, boosters]).reset_index().set_index(['course', 'location_id', 'date'])    

    total_vaccinations, efficacy = get_total_vaccinations_and_efficacy(vaccinations)
    waning = build_waning_dist(efficacy)
    etas = compute_eta(total_vaccinations, efficacy, waning)
    total_vaccinations = pivot_risk_group(total_vaccinations).reset_index()
    etas = pivot_risk_group(etas)

    etas_final = []
    for efficacy_type in ['immune', 'protected']:
        eta = etas[[c for c in etas.columns if efficacy_type in c]]
        eta.columns = ['_'.join([x for x in c.split('_') if x != efficacy_type]) for c in eta.columns]
        eta['efficacy_type'] = efficacy_type
        eta = eta.reset_index()
        etas_final.append(eta)
    etas_final = pd.concat(etas_final)

    summaries = data[['cumulative_all_effective', 'cumulative_all_vaccinated', 
                      'cumulative_all_fully_vaccinated', 
                      'hr_vaccinated', 'lr_vaccinated']]
    series_acceptance = load_series_acceptance(series_path)
    point_acceptance = load_point_acceptance(point_path).reindex(series_acceptance.index, level='location_id')
    summaries = pd.concat([summaries, series_acceptance, point_acceptance], axis=1).reset_index()
    return total_vaccinations, etas_final, summaries


def load_series_acceptance(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index(['location_id', 'date']).sort_index().smooth_combined_yes.rename('vaccine_acceptance')
    return data


def load_point_acceptance(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.set_index('location_id').sort_index().smooth_combined_yes.rename('vaccine_acceptance_point')
    return data    


def extract_vaccine_categories(data: pd.DataFrame, population_group: str):
    column_map = {
        'unprotected': 'unprotected',
        'effective_protected_wildtype': 'non_escape_protected',
        'effective_protected_variant': 'escape_protected',
        'effective_wildtype': 'non_escape_immune',
        'effective_variant': 'escape_immune',
    }
    columns = []
    for from_name, to_name in column_map.items():
        columns.append(data[f'{population_group}_{from_name}'].rename(f'vaccinations_{to_name}_{population_group}'))
    data = pd.concat(columns, axis=1).fillna(0.)
    data[f'vaccinations_omega_protected_{population_group}'] = 0.
    data[f'vaccinations_omega_immune_{population_group}'] = 0.
    return data


def get_total_vaccinations_and_efficacy(vaccinations: pd.DataFrame):
    """Create total vaccinations and average efficacy of delivered doses by variant."""
    ordered_efficacy_cols = ['omega_immune', 'escape_immune', 'non_escape_immune',
                             'omega_protected', 'escape_protected', 'non_escape_protected']
    v, e = [], []
    for risk_group in RISK_GROUP_NAMES:
        df = vaccinations.filter(like=risk_group).reset_index()
        df['risk_group'] = risk_group
        df = df.set_index(['course', 'location_id', 'risk_group', 'date'])
        total_vaccinations = df.sum(axis=1).rename(f'vaccinations')
        eps = []
        for i, eps_type in enumerate(ordered_efficacy_cols):
            cols = [f'vaccinations_{efficacy}_{risk_group}' for efficacy in ordered_efficacy_cols[:i + 1]]
            eps.append((df[cols].sum(axis=1) / total_vaccinations).fillna(0).rename(f'{eps_type}'))
        eps = pd.concat(eps, axis=1)
        v.append(total_vaccinations)
        e.append(eps)

    return pd.concat(v), remap_efficacy(pd.concat(e))


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


def build_waning_dist(efficacy: pd.DataFrame, waning_params: Tuple[float, int, float, int] = WANING_PARAMS):
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


def compute_eta(total_vaccinations, efficacy, waning):
    total_vaccinations = total_vaccinations.sort_index()
    efficacy = efficacy.sort_index()
    weighted_vaccinations = efficacy.mul(total_vaccinations, axis=0)
    location_ids = total_vaccinations.reset_index().location_id.unique()
    args = []
    for location_id, course, risk_group in itertools.product(location_ids, [1, 2], RISK_GROUP_NAMES):
        idx = (course, location_id, risk_group)
        args.append((
            course,
            location_id,
            risk_group,
            total_vaccinations.loc[idx],
            weighted_vaccinations.loc[idx],
            waning.loc[idx]
        ))

    with multiprocessing.Pool(12) as pool, np.errstate(divide='ignore'):
        etas = list(tqdm.tqdm(pool.imap(_compute_eta, args), total=len(args)))
    etas = pd.concat(etas)
    return etas


def _compute_eta(args):
    course, location_id, risk_group, v, ve, w = args    
    dates = ve.index
    columns = ve.columns
    v, ve, w = v.values, ve.values, w.values

    eta = np.zeros(ve.shape)
    for t in range(1, v.size + 1):
        eta[t - 1] = (ve[:t][::-1] * w[:t]).sum(axis=0) / v[:t].sum()
    eta = pd.DataFrame(eta, columns=columns, index=dates)
    eta['course'] = course
    eta['risk_group'] = risk_group
    eta['location_id'] = location_id
    eta = eta.reset_index().set_index(['course', 'risk_group', 'location_id', 'date'])
    eta = eta.bfill().fillna(0.)
    return eta


def pivot_risk_group(data):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    new_levels = [n for n in data.index.names if n != 'risk_group'] + ['risk_group']
    data = data.reorder_levels(new_levels).sort_index().unstack()
    data.columns = ['_'.join(levels) for levels in data.columns]
    return data

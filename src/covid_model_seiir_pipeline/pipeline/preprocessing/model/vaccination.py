import itertools
from typing import Dict, List

import numba
import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)

logger = cli_tools.task_performance_logger


def make_uptake_square(uptake: pd.DataFrame) -> pd.DataFrame:
    courses = uptake.vaccine_course.unique()
    location_ids = uptake.location_id.unique()
    risk_groups = uptake.risk_group.unique()
    date = pd.date_range(uptake.date.min(), uptake.date.max())
    idx_names = ['vaccine_course', 'location_id', 'risk_group', 'date']
    idx = pd.MultiIndex.from_product([courses, location_ids, risk_groups, date], names=idx_names)
    uptake = uptake.set_index(idx_names).sort_index()
    duplicates = uptake.index.duplicated()
    if np.any(duplicates):
        logger.warning('Duplicates found in uptake dataset')
        uptake = uptake.loc[~duplicates]
    uptake = uptake.reindex(idx).fillna(0.)
    return uptake


def map_variants(efficacy: pd.DataFrame) -> pd.DataFrame:
    efficacy_map = {
        'alpha': 'ancestral',
        'beta': 'delta',
        'gamma': 'delta',
        'other': 'delta',
        'omicron': 'omicron',
        'omega': 'omicron',
    }
    for target_variant, similar_variant in efficacy_map.items():
        efficacy[target_variant] = efficacy[similar_variant]
    return efficacy


def rescale_to_proportion(waning: pd.DataFrame) -> pd.DataFrame:
    max_efficacy = (waning
                    .groupby(['endpoint', 'brand'])
                    .max()
                    .reindex(waning.index))
    waning = (waning / max_efficacy).fillna(0.)
    return waning


def get_infection_endpoint_brand_specific_waning(waning: pd.DataFrame,
                                                all_brands: List[str]) -> pd.DataFrame:
    waning_map = {
        'BNT-162': waning.loc[('infection', 'pfi')],
        'Moderna': waning.loc[('infection', 'mod')],
        'AZD1222': waning.loc[('infection', 'ast')],
        'Janssen': waning.loc[('infection', 'jan')],
    }
    default_waning = pd.concat(waning_map.values(), axis=1).mean(axis=1).rename('value')
    brand_specific_waning = _get_brand_specific_waning(waning_map, default_waning, all_brands)
    brand_specific_waning['endpoint'] = 'infection'
    brand_specific_waning = (brand_specific_waning
                             .reset_index()
                             .set_index(['endpoint', 'brand', 'days'])
                             .sort_index())
    return brand_specific_waning


def get_severe_endpoint_brand_specific_waning(waning: pd.DataFrame,
                                              all_brands: List[str]) -> pd.DataFrame:
    waning_map = {
        'BNT-162': waning.loc[('severe_disease', 'pfi')],
        'Moderna': waning.loc[('severe_disease', 'mod')],
        'AZD1222': waning.loc[('severe_disease', 'ast')],
        'Janssen': waning.loc[('severe_disease', 'jan')],
    }
    default_waning = pd.concat(waning_map.values(), axis=1).mean(axis=1).rename('value')
    brand_specific_waning = _get_brand_specific_waning(waning_map, default_waning, all_brands)
    brand_specific_waning['endpoint'] = 'severe_disease'
    brand_specific_waning = (brand_specific_waning
                             .reset_index()
                             .set_index(['endpoint', 'brand', 'days'])
                             .sort_index())
    return brand_specific_waning


def _get_brand_specific_waning(waning_map: Dict[str, pd.Series],
                               default_waning: pd.Series,
                               all_brands: List[str]):
    out = []
    for brand in all_brands:
        brand_waning = waning_map.get(brand, default_waning)
        brand_waning = _coerce_week_index_to_day_index(brand_waning).to_frame()
        brand_waning['brand'] = brand
        out.append(brand_waning)
    waning = pd.concat(out).reset_index().set_index(['brand', 'days']).sort_index()
    return waning


def _coerce_week_index_to_day_index(data: pd.Series) -> pd.Series:
    t_wks = 7 * data.index
    t_days = np.arange(0, t_wks.max())
    data = pd.Series(
        np.interp(t_days, t_wks, data),
        index=pd.Index(t_days, name='days'),
        name='value'
    )
    data = data.reindex(np.arange(0, 1500)).fillna(0.)
    return data


def build_waning_efficacy(efficacy: pd.DataFrame, waning: pd.DataFrame) -> pd.DataFrame:
    waning_efficacy = efficacy.join(waning)
    col_names = waning_efficacy.index.names
    waning_efficacy = (waning_efficacy
                       .loc[:, efficacy.columns]
                       .mul(waning_efficacy.value, axis=0)
                       .stack()
                       .reset_index())
    waning_efficacy.columns = col_names + ['variant', 'value']
    waning_efficacy = (waning_efficacy
                       .set_index(['endpoint', 'vaccine_course', 'variant', 'days', 'brand'])
                       .value
                       .sort_index()
                       .unstack())
    waning_efficacy.columns.name = None
    return waning_efficacy


def build_eta_calc_arguments(vaccine_uptake: pd.DataFrame,
                             waning_efficacy: pd.DataFrame,
                             progress_bar: bool) -> List:
    location_ids = vaccine_uptake.reset_index().location_id.unique()
    groups = itertools.product(location_ids, [1, 2], ['hr', 'lr'])
    eta_args = []

    infection_efficacy = waning_efficacy.loc['infection']
    severe_disease_efficacy = waning_efficacy.loc['severe_disease']

    for location_id, vaccine_course, risk_group in tqdm.tqdm(list(groups), disable=not progress_bar):
        group_uptake = vaccine_uptake.loc[(vaccine_course, location_id, risk_group)]
        group_infection_efficacy = infection_efficacy.loc[vaccine_course]
        group_severe_disease_efficacy = severe_disease_efficacy.loc[vaccine_course]

        eta_args.append([
            'infection',
            location_id,
            vaccine_course,
            risk_group,
            group_uptake,
            group_infection_efficacy,
            pd.DataFrame(0., columns=group_infection_efficacy.columns, index=group_infection_efficacy.index),
        ])
        eta_args.append([
            'severe_disease',
            location_id,
            vaccine_course,
            risk_group,
            group_uptake,
            group_severe_disease_efficacy,
            group_infection_efficacy,
        ])

    return eta_args


def compute_eta(args: List) -> pd.DataFrame:
    endpoint, location_id, vaccine_course, risk_group, uptake, efficacy_target, efficacy_base = args
    variants = efficacy_target.reset_index().variant.unique()
    dates = uptake.index
    u = uptake.values

    eta = np.zeros((len(dates), len(variants)))
    for j, variant in enumerate(variants):
        e_t = efficacy_target.loc[variant].values[:len(u)]
        e_b = efficacy_base.loc[variant].values[:len(u)]
        _compute_variant_eta(u, e_t, e_b, j, eta)

    eta = pd.DataFrame(eta, columns=variants, index=dates)
    eta['endpoint'] = endpoint
    eta['location_id'] = location_id
    eta['vaccine_course'] = vaccine_course
    eta['risk_group'] = risk_group

    eta = eta.reset_index().set_index(['endpoint', 'location_id', 'date', 'vaccine_course', 'risk_group'])
    eta = eta.bfill().fillna(0.)
    return eta


@numba.njit
def _compute_variant_eta(u, e_t, e_b, j, eta):
    for t in range(1, u.shape[0] + 1):
        total = u[:t].sum()
        if total:
            eta[t - 1, j] = (u[:t][::-1] * (1 - (1 - e_t[:t]) / (1 - e_b[:t]))).sum() / total
        else:
            eta[t - 1, j] = 0.
    return eta


def compute_natural_waning(waning: pd.DataFrame) -> pd.DataFrame:
    # Other is an average of all vaccine waning, which is also what we want for natural waning.
    natural_waning_infection = waning.loc[('infection', 'Other')].reset_index()
    natural_waning_infection['endpoint'] = 'infection'
    natural_waning_case = natural_waning_infection.copy()
    natural_waning_case['endpoint'] = 'case'
    natural_waning_severe_disease = waning.loc[('severe_disease', 'Other')].reset_index()
    natural_waning_death = natural_waning_severe_disease.copy()
    natural_waning_death['endpoint'] = 'death'
    natural_waning_admission = natural_waning_severe_disease.copy()
    natural_waning_admission['endpoint'] = 'admission'
    natural_waning = (pd.concat([natural_waning_infection, 
                                 natural_waning_case, 
                                 natural_waning_death, 
                                 natural_waning_admission])
                      .set_index(['days', 'endpoint'])
                      .sort_index()
                      .unstack())
    natural_waning.columns = natural_waning.columns.droplevel().rename(None)
    return natural_waning

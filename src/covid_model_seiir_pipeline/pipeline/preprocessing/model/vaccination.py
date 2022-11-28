import itertools
from typing import Dict, List

import numba
import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    parallel,
)

logger = cli_tools.task_performance_logger


def make_uptake_square(uptake: pd.DataFrame) -> pd.DataFrame:
    courses = [1, 2, 3, 4 ] #uptake.vaccine_course.unique()
    location_ids = uptake.location_id.unique()
    risk_groups = uptake.risk_group.unique()
    date = pd.date_range(uptake.date.min(), uptake.date.max())
    add_fourth_dose = 4 in uptake.vaccine_course.unique() and 'targeted' in uptake

    name_map = {
        'BNT.162': 'BNT-162',
        'BNT-162': 'BNT-162',
        'BNT 162': 'BNT-162',
        'Moderna': 'Moderna',
        'AZD1222': 'AZD1222',
        'Janssen': 'Janssen',
        'Sputnik.V': 'Sputnik V',
        'Sputnik V': 'Sputnik V',
        'Novavax': 'Novavax',
        'CoronaVac': 'CoronaVac',
        'CNBG.Wuhan': 'CNBG Wuhan',
        'CNBG Wuhan': 'CNBG Wuhan',
        'Tianjin.CanSino': 'Tianjin CanSino',
        'Tianjin CanSino': 'Tianjin CanSino',
        'Covaxin': 'Covaxin',
        'mRNA.Vaccine': 'mRNA Vaccine',
        'mRNA Vaccine': 'mRNA Vaccine',
        'Other': 'Other',
        'targeted': 'targeted',
    }

    idx_names = ['vaccine_course', 'location_id', 'risk_group', 'date']
    vax_names = sorted(list(set(name_map.values()).difference(['targeted'])))
    idx = pd.MultiIndex.from_product([courses, location_ids, risk_groups, date], names=idx_names)
    uptake = uptake.set_index(idx_names).sort_index().rename(columns=name_map)
    if add_fourth_dose:
        uptake.loc[[4], 'mRNA Vaccine'] = uptake.loc[[4], 'targeted']
    uptake = uptake.loc[:, vax_names]
    duplicates = uptake.index.duplicated()
    if np.any(duplicates):
        logger.warning('Duplicates found in uptake dataset')
        uptake = uptake.loc[~duplicates]
    uptake = uptake.reindex(idx).fillna(0.)
    return uptake


def build_eta_calc_arguments(vaccine_uptake: pd.DataFrame,
                             waning_efficacy: pd.DataFrame,
                             progress_bar: bool) -> List:
    location_ids = vaccine_uptake.reset_index().location_id.unique()
    courses = vaccine_uptake.reset_index().vaccine_course.unique()
    risk_groups = vaccine_uptake.reset_index().risk_group.unique()
    groups = itertools.product(location_ids, courses, risk_groups)
    eta_args = []
    waning_efficacy = waning_efficacy.stack().reset_index().rename(columns={'level_4': 'variant', 0: 'value'})
    waning_efficacy = waning_efficacy.set_index(['endpoint', 'vaccine_course', 'variant', 'days', 'brand']).unstack()
    waning_efficacy.columns.name = None
    
    infection_efficacy = waning_efficacy.loc['infection']
    case_efficacy = waning_efficacy.loc['case']
    admission_efficacy = waning_efficacy.loc['admission']
    death_efficacy = waning_efficacy.loc['death']

    for location_id, vaccine_course, risk_group in tqdm.tqdm(list(groups), disable=not progress_bar):
        try:
            group_uptake = vaccine_uptake.loc[(vaccine_course, location_id, risk_group)]
        except KeyError:
            logger.warning(f'Missing uptake for location: {location_id}, vaccine course: {vaccine_course}, risk group: {risk_group}')
            continue
        group_infection_efficacy = infection_efficacy.loc[vaccine_course]
        group_case_efficacy = case_efficacy.loc[vaccine_course]
        group_admission_efficacy = admission_efficacy.loc[vaccine_course]
        group_death_efficacy = death_efficacy.loc[vaccine_course]

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
            'case',
            location_id,
            vaccine_course,
            risk_group,
            group_uptake,
            group_case_efficacy,
            group_infection_efficacy,
        ])
        eta_args.append([
            'admission',
            location_id,
            vaccine_course,
            risk_group,
            group_uptake,
            group_admission_efficacy,
            group_infection_efficacy,
        ])
        eta_args.append([
            'death',
            location_id,
            vaccine_course,
            risk_group,
            group_uptake,
            group_death_efficacy,
            group_infection_efficacy,
        ])

    return eta_args


def build_vaccine_risk_reduction(eta_args: List,
                                 num_cores: int,
                                 progress_bar: bool):
    etas = parallel.run_parallel(
        compute_eta,
        arg_list=eta_args,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    etas = (pd.concat(etas)
            .reorder_levels(
        ['vaccine_course', 'endpoint', 'risk_group', 'location_id', 'date'])
            .sort_index())
    if etas.values.min() < 0.:
        raise ValueError('etas less than 0.')
    if etas.values.max() > 1.:
        raise ValueError('etas over 1.')
    etas_unvax = etas.loc[1].copy()
    etas_unvax.loc[:, :] = 0.
    etas_unvax['vaccine_course'] = 0
    etas_unvax = (etas_unvax
                  .set_index('vaccine_course', append=True)
                  .reorder_levels(
        ['vaccine_course', 'endpoint', 'risk_group', 'location_id', 'date']))
    etas = (pd.concat([etas, etas_unvax])
            .reorder_levels(
        ['endpoint', 'location_id', 'date', 'vaccine_course', 'risk_group']))
    courses = etas.reset_index().vaccine_course.unique().tolist()
    risk_groups = etas.reset_index().risk_group.unique().tolist()
    etas = etas.unstack().unstack()
    etas.columns = [f'course_{vaccine_course}_{variant}_{risk_group}'
                    for variant, risk_group, vaccine_course in etas.columns]
    extras_cols = [f'course_{c}_none_{g}' for c, g in
                   itertools.product(courses, risk_groups)]
    etas.loc[:, extras_cols] = 0.

    risk_reductions = etas.reorder_levels(['location_id', 'date', 'endpoint']).sort_index().unstack()
    risk_reductions.columns = [f'{c[:-3]}_{e}_{c[-2:]}' for c, e in risk_reductions.columns]
    return risk_reductions


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

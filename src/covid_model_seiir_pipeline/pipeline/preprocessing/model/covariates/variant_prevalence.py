from collections import defaultdict
import itertools
from pathlib import Path
from typing import Dict

from covid_shared import paths as paths
from loguru import logger
import pandas as pd

from covid_input_seir_covariates.utilities import CovariateGroup


COVARIATE_NAMES = (
    'variant_prevalence',
)

DEFAULT_OUTPUT_ROOT = paths.VARIANT_OUTPUT_ROOT


def get_covariates(covariates_root: Path) -> CovariateGroup:
    covariate_data = {}
    for scenario in ['reference', 'worse']:
        covariate_data[scenario] = _load_variant_data(
            covariates_root / f'variant_{scenario}.csv'
        )
    return {'variant_prevalence': ({}, covariate_data)}


def _load_variant_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date', 'variant']).prevalence
            .sort_index()
            .unstack())

    variant_map = {
        'alpha': ['B117'],
        'beta': ['B1351'],
        'gamma': ['P1'],
        'delta': ['B16172'],
        'other': ['B1621', 'C37', 'C12'],
        'ancestral': ['wild_type'],
    }
    drop = []
    all_variants = [lineage for lineages in variant_map.values() for lineage in lineages] + drop
    missing_variants = set(all_variants).difference(data.columns)
    if missing_variants:
        raise ValueError(f'Missing variants {missing_variants}')
    extra_variants = set(data.columns).difference(all_variants)
    if extra_variants:
        raise ValueError(f'Unknown variants {extra_variants}')
    if drop:
        logger.warning(f'Dropping variants: {drop}')

    for var_name, lineages in variant_map.items():
        data[var_name] = data[lineages].sum(axis=1)
    data = data[list(variant_map)]
    data['omega'] = 0.
    if (data.sum(axis=1) < 1 - 1e-5).any():
        raise ValueError("Variant prevalence sums to less than 1 for some location-dates.")
    if (data.sum(axis=1) > 1 + 1e-5).any():
        raise ValueError("Variant prevalence sums to more than 1 for some location-dates.")

    return data.reset_index()

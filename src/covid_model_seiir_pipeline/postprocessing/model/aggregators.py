"""Postprocessing aggregation strategies."""
from collections import defaultdict
from typing import List

from loguru import logger
import pandas as pd


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(axis=1).rename('mean')
    upper = data.quantile(.975, axis=1).rename('upper')
    lower = data.quantile(.025, axis=1).rename('lower')
    return pd.concat([mean, upper, lower], axis=1)


def sum_aggregator(measure_data: pd.DataFrame, hierarchy: pd.DataFrame, _population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates most-detailed locations to locations in the hierarchy by sum.

    The ``_population`` argument is here for api consistency and is not used.

    """
    if 'observed' in measure_data.index.names:
        measure_data = measure_data.reset_index(level='observed')

    levels = sorted(hierarchy['level'].unique().tolist())
    missing_map = defaultdict(list)
    data_locs = _get_modeled_locs(measure_data, hierarchy)
    for level in reversed(levels[1:]):  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]

        for parent_id, children in locs_at_level.groupby('parent_id'):
            if parent_id in 
            child_locs = children.location_id.tolist()
            modeled_child_locs = list(set(child_locs).intersection(data_locs))

            aggregate = measure_data.loc[modeled_child_locs].groupby(level='date').sum()
            aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])

            measure_data = measure_data.append(aggregate)
            data_locs.append(parent_id)

            for child_loc in child_locs:
                if child_loc in missing_map:
                    missing_map[parent_id].extend(missing_map[child_loc])
                elif child_loc not in modeled_child_locs:
                    missing_map[parent_id].append(child_loc)

    # We'll call any aggregate with at least one observed point observed.
    if 'observed' in measure_data.columns:
        measure_data.loc[measure_data['observed'] > 0, 'observed'] = 1
        measure_data = measure_data.set_index('observed', append=True)
    # TODO: Figure out what to do with the missing map.
    return measure_data.sort_index()


def mean_aggregator(measure_data: pd.DataFrame, hierarchy: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates most-detailed locations to locations in the hierarchy by
    population-weighted mean.

    """
    if 'observed' in measure_data.index.names:
        measure_data = measure_data.reset_index(level='observed')
    # Get all age/sex population and append to the data.
    population = _collapse_population(population)
    measure_columns = measure_data.columns
    measure_data = measure_data.join(population)
    
    modeled_locs = _get_modeled_locs(measure_data, hierarchy)
    # We'll work in the space of pop*measure where aggregation is just a sum.
    weighted_measure_data = measure_data.loc[modeled_locs]
    weighted_measure_data[measure_columns] = measure_data[measure_columns].mul(measure_data['population'], axis=0)

    levels = sorted(hierarchy['level'].unique().tolist())
    missing_map = defaultdict(list)
    data_locs = modeled_locs[:]
    measure_data = measure_data.loc[data_locs]
    for level in reversed(levels[1:]):  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]
        for parent_id, children in locs_at_level.groupby('parent_id'):
            child_locs = children.location_id.tolist()
            modeled_child_locs = list(set(child_locs).intersection(data_locs))

            if 'date' in weighted_measure_data.index.names:
                aggregate = weighted_measure_data.loc[modeled_child_locs].groupby(level='date').sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])
            else:
                aggregate = weighted_measure_data.loc[modeled_child_locs].sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id']).unstack()

            weighted_measure_data = weighted_measure_data.append(aggregate)
            aggregate = aggregate[measure_columns].div(aggregate['population'], axis=0)
            measure_data = measure_data.append(aggregate)

            data_locs.append(parent_id)

            for child_loc in child_locs:
                if child_loc in missing_map:
                    missing_map[parent_id].extend(missing_map[child_loc])
                elif child_loc not in modeled_child_locs:
                    missing_map[parent_id].append(child_loc)

    measure_data = measure_data.drop(columns='population')
    # We'll call any aggregate with at least one observed point observed.
    if 'observed' in measure_data.columns:
        measure_data.loc[measure_data['observed'] > 0, 'observed'] = 1
        measure_data = measure_data.set_index('observed', append=True)
    # TODO: Figure out what to do with the missing map.
    return measure_data.sort_index()


def _collapse_population(population_data: pd.DataFrame) -> pd.DataFrame:
    """Collapse the larger population table to all age and sex population."""
    all_sexes = population_data.sex_id == 3
    all_ages = population_data.age_group_id == 22
    population_data = population_data.loc[all_ages & all_sexes, ['location_id', 'population']]
    population_data = population_data.set_index('location_id')['population']
    return population_data


def _get_modeled_locs(measure_data: pd.DataFrame, hierarchy: pd.DataFrame) -> List[int]:
    modeled_locs = measure_data.reset_index().location_id.unique().tolist()

    non_most_detailed_h = hierarchy.loc[hierarchy.most_detailed == 0, 'location_id'].tolist()
    non_most_detailed_m = list(set(modeled_locs).intersection(non_most_detailed_h))
    if non_most_detailed_m:
        logger.warning(f'Non most-detailed locations {non_most_detailed_h} found in data. '
                        'These locations will not be aggregated.')
    return modeled_locs

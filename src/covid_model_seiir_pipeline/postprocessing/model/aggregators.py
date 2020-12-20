"""Postprocessing aggregation strategies."""
from collections import defaultdict

import pandas as pd


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(axis=1).rename('mean')
    upper = data.quantile(.975, axis=1).rename('upper')
    lower = data.quantile(.025, axis=1).rename('lower')
    return pd.concat([mean, upper, lower], axis=1)


def sum_aggregator(measure_data: pd.DataFrame, hierarchy: pd.DataFrame, _population: pd.DataFrame) -> pd.DataFrame, Dict[int, List[int]]:
    """Aggregates most-detailed locations to locations in the hierarchy by sum.

    The ``_population`` argument is here for api consistency and is not used.

    """
    measure_data = measure_data.reset_index()
    modeled_locs = measure_data.location_id.unique().tolist()

    non_most_detailed_h = hierarchy.loc[hierarchy.most_detailed == 0, 'location_id'].tolist()
    non_most_detailed_m = list(set(modeled_locs).intersection(non_most_detailed_h))

    assert not non_most_detailed_m, 'Why are we modeling aggregate locations? Maybe a bad hierarchy.'

    measure_data = measure_data.set_index(['location_id', 'date'])
    levels = sorted(hierarchy['level'].unique().tolist())
    missing_map = defaultdict(list)
    data_locs = modeled_locs[:]
    for level in levels[::-1]:  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]

        for parent_id, children in locs_at_level.groupby('parent_id'):
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
        measure_data.loc[measure_data['observed'] >= 1, 'observed'] = 1
        measure_data = measure_data.set_index('observed', append=True)
    # TODO: Figure out what to do with the missing map.
    return measure_data


def mean_aggregator(measure_data: pd.DataFrame, hierarchy: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates most-detailed locations to locations in the hierarchy by
    population-weighted mean.

    """
    # Get all age/sex population and append to the data.
    population = _collapse_population(population, hierarchy)
    measure_columns = measure_data.columns
    measure_data = measure_data.join(population)

    # We'll work in the space of pop*measure where aggregation is just a sum.
    weighted_measure_data = measure_data.loc[hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].tolist()]
    weighted_measure_data[measure_columns] = measure_data[measure_columns].mul(measure_data['population'], axis=0)

    # Some covariates are time-invariant
    if 'date' in weighted_measure_data.index.names:
        global_aggregate = weighted_measure_data.groupby(level='date').sum()
        global_aggregate = pd.concat({1: global_aggregate}, names=['location_id'])
    else:
        global_aggregate = weighted_measure_data.sum()
        global_aggregate = pd.concat({1: global_aggregate}, names=['location_id']).unstack()
    # Invert the weighting.  Pop column is also aggregated now.
    global_aggregate = global_aggregate[measure_columns].div(global_aggregate['population'], axis=0)
    measure_data = measure_data.append(global_aggregate)

    # Do the same thing again to produce national aggregates.
    national_level = 3
    subnational_levels = sorted([level for level in hierarchy['level'].unique().tolist() if level > national_level])
    for level in subnational_levels[::-1]:  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]
        for parent_id, children in locs_at_level.groupby('parent_id'):
            child_locs = children.location_id.tolist()
            if 'date' in weighted_measure_data.index.names:
                aggregate = weighted_measure_data.loc[child_locs].groupby(level='date').sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])

            else:
                aggregate = weighted_measure_data.loc[child_locs].sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id']).unstack()

            weighted_measure_data = weighted_measure_data.append(aggregate)
            aggregate = aggregate[measure_columns].div(aggregate['population'], axis=0)
            measure_data = measure_data.append(aggregate)
    measure_data = measure_data.drop(columns='population')
    return measure_data


def _collapse_population(population_data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Collapse the larger population table to all age and sex population."""
    most_detailed = population_data.location_id.isin(
        hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].tolist()
    )
    all_sexes = population_data.sex_id == 3
    all_ages = population_data.age_group_id == 22
    population_data = population_data.loc[most_detailed & all_ages & all_sexes, ['location_id', 'population']]
    population_data = population_data.set_index('location_id')['population']
    return population_data

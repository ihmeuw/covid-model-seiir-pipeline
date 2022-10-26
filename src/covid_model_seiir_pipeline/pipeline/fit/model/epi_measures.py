from typing import List

import numpy as np
import pandas as pd


def filter_and_format_epi_measures(
    epi_measures: pd.DataFrame,
    mortality_scalar: pd.Series,
    mr_hierarchy: pd.DataFrame,
    pred_hierarchy: pd.DataFrame,
    measure_lag: int,
    max_lag: int,
    variant_prevalence: pd.DataFrame,
    epi_exclude_variants: List[str],
    measure: str = None,
) -> pd.DataFrame:
    if measure:
        epi_measures = drop_locations_for_measure_model(epi_measures, measure, pred_hierarchy)

    if epi_exclude_variants:
        epi_measures = exclude_epi_data_by_variant(
            epi_measures=epi_measures,
            pred_hierarchy=pred_hierarchy,
            variant_prevalence=variant_prevalence.loc[:, epi_exclude_variants].sum(axis=1).rename('variant_prevalence'),
            measure_lag=measure_lag,
        )

    epi_measures = format_epi_measures(epi_measures, mr_hierarchy, pred_hierarchy,
                                       mortality_scalar, max_lag)

    if measure:
        epi_measures = enforce_epi_threshold(epi_measures, measure, mortality_scalar)

    return epi_measures


def drop_locations_for_measure_model(
    epi_measures: pd.DataFrame,
    measure: str,
    hierarchy: pd.DataFrame,
) -> pd.DataFrame:
    """Removes a set of locations from a specific measure model."""
    drop_location_ids = {
        # if it's a parent location, will apply to all children as well
        'death': [
            44533,  # Mainland China
            151,    # Qatar
            4840,   # Andaman and Nicobar Islands
            60896,  # Dadra and Nagar Haveli and Daman and Diu
            4858,   # Lakshadweep
            186,    # Seychelles
            169,    # Central African Republic
        ],
        'case': [],
        'admission': [
            97,  # Argentina
        ],
    }[measure]

    drop_location_ids = [
        (hierarchy.loc[hierarchy['path_to_top_parent']
                  .apply(lambda x: str(loc_id) in x.split(',')), 'location_id']
                  .to_list())
        for loc_id in drop_location_ids
    ]
    if drop_location_ids:
        drop_location_ids = np.unique((np.hstack(drop_location_ids))).tolist()
        epi_measures = epi_measures.drop(drop_location_ids)
    return epi_measures


def exclude_epi_data_by_variant(
    epi_measures: pd.DataFrame,
    pred_hierarchy: pd.DataFrame,
    variant_prevalence: pd.DataFrame,
    measure_lag: int,
) -> pd.DataFrame:
    md_location_ids = pred_hierarchy.loc[pred_hierarchy['most_detailed'] == 1, 'location_id'].astype(str).to_list()
    epi_measures = epi_measures.query(f'location_id in [{", ".join(md_location_ids)}]')

    variant_prevalence = variant_prevalence.loc[variant_prevalence > 0.01]
    drop_dates = variant_prevalence.reset_index('date').groupby('location_id')['date'].min()
    drop_dates += pd.Timedelta(days=measure_lag)

    epi_measures = epi_measures.groupby('location_id').apply(lambda x: _exclude_epi_data_by_variant(x, drop_dates))

    return epi_measures


def _exclude_epi_data_by_variant(location_data: pd.DataFrame, drop_dates: pd.Series):
    location_id = location_data.index.get_level_values('location_id').unique().item()
    return (location_data
            .loc[:, :drop_dates.loc[location_id], :]
            .reset_index('location_id', drop=True))


def enforce_epi_threshold(
    epi_measures: pd.DataFrame,
    measure: str,
    mortality_scalar: pd.Series
) -> pd.DataFrame:
    thresholds = {
        'case': 100,
        'admission': 10,
        'death': 10,
    }
    cumul_cols = {
        'case': 'cumulative_cases',
        'admission': 'cumulative_hospitalizations',
        'death': 'cumulative_deaths',
    }
    max_reported = (epi_measures
                    .loc[:, cumul_cols[measure]]
                    .groupby('location_id')
                    .max())
    if measure == 'death':
        threshold = mortality_scalar.groupby('location_id').mean() * thresholds[measure]
        threshold = threshold.loc[max_reported.index]
    else:
        threshold = thresholds[measure]

    above_threshold_locations = max_reported.loc[max_reported >= threshold].index.to_list()

    return epi_measures.loc[above_threshold_locations]


def format_epi_measures(
    epi_measures: pd.DataFrame,
    mr_hierarchy: pd.DataFrame,
    pred_hierarchy: pd.DataFrame,
    mortality_scalars: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    deaths = (epi_measures['cumulative_deaths'] * mortality_scalars).rename('cumulative_deaths').dropna()
    deaths = _format_measure(deaths, mr_hierarchy, pred_hierarchy)
    cases = _format_measure(epi_measures['cumulative_cases'], mr_hierarchy, pred_hierarchy)
    admissions = _format_measure(epi_measures['cumulative_hospitalizations'], mr_hierarchy, pred_hierarchy)
    epi_measures = pd.concat([deaths, cases, admissions], axis=1).reset_index()

    locations = epi_measures.location_id.unique()
    dates = epi_measures.date
    global_date_range = pd.date_range(dates.min() - pd.Timedelta(days=max_lag), dates.max())
    square_idx = pd.MultiIndex.from_product((locations, global_date_range),
                                            names=['location_id', 'date']).sort_values()
    epi_measures = epi_measures.set_index(['location_id', 'date']).reindex(square_idx).sort_index()

    return epi_measures


def _format_measure(data: pd.Series,
                    mr_hierarchy: pd.DataFrame,
                    pred_hierarchy: pd.DataFrame) -> pd.DataFrame:
    measure = str(data.name).split('_')[-1]
    data = data.reset_index()
    extra_locations = set(pred_hierarchy.loc[pred_hierarchy['most_detailed'] == 1, 'location_id'])
    extra_locations = list(extra_locations.difference(mr_hierarchy['location_id']))
    extra_data = data.loc[data['location_id'].isin(extra_locations)]

    data = aggregate_data_from_md(data, mr_hierarchy, f'cumulative_{measure}')
    data = (data
            .append(extra_data.loc[:, data.columns])
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    data = make_daily_and_smoothed_daily(data, measure)
    return data


def make_daily_and_smoothed_daily(data: pd.DataFrame, measure: str) -> pd.DataFrame:
    data[f'daily_{measure}'] = (data
                                .groupby(['location_id'])[f'cumulative_{measure}']
                                .diff()
                                .fillna(data[f'cumulative_{measure}']))
    data = data.dropna().set_index(['location_id', 'date']).sort_index()

    data[f'smoothed_daily_{measure}'] = (data[f'daily_{measure}']
                                         .groupby('location_id')
                                         .apply(lambda x: x.clip(0, np.inf)
                                                .rolling(window=7, min_periods=7, center=True)
                                                .mean()))
    return data


def aggregate_data_from_md(data: pd.DataFrame, hierarchy: pd.DataFrame, agg_var: str,
                           check_for_rates: bool = True, require_all_chilren: bool = True) -> pd.DataFrame:
    if check_for_rates:
        if data[agg_var].max() <= 1:
            raise ValueError(f'Data in {agg_var} looks like rates - need counts for aggregation.')

    data = data.copy()

    is_md = hierarchy['most_detailed'] == 1
    md_location_ids = hierarchy.loc[is_md, 'location_id'].to_list()
    parent_location_ids = hierarchy.loc[~is_md, 'location_id'].to_list()

    md_data = data.loc[data['location_id'].isin(md_location_ids)]

    md_child_ids_lists = [(hierarchy
                           .loc[is_md & (
        hierarchy['path_to_top_parent'].apply(lambda x: str(parent_location_id) in x.split(','))),
                                'location_id']
                           .to_list()) for parent_location_id in parent_location_ids]
    parent_children_pairs = list(zip(parent_location_ids, md_child_ids_lists))

    parent_data = [aggregate(md_data.loc[md_data['location_id'].isin(md_child_ids)],
                             parent_id, md_child_ids, agg_var, require_all_chilren)
                   for parent_id, md_child_ids in parent_children_pairs]
    data = pd.concat([md_data] + parent_data)

    return data.reset_index(drop=True)


def aggregate(data: pd.DataFrame, parent_id: int, md_child_ids: List[int], agg_var: str,
              require_all_chilren: bool = True) -> pd.DataFrame:
    # not most efficient to go from md each time, but safest since dataset is not square (and not a ton of data)
    draw_level = 'draw' in data.columns
    if data.empty:
        if draw_level:
            return data.loc[:, ['draw', 'location_id', 'date', agg_var]]
        else:
            return data.loc[:, ['location_id', 'date', agg_var]]
    else:
        # REQUIRE ALL CHILD LOCATIONS
        if draw_level:
            data = data.groupby(['draw', 'date'])[agg_var].agg(['sum', 'count'])
            data['count'] = data.groupby('date')['count'].transform(min)
        else:
            data = data.groupby('date')[agg_var].agg(['sum', 'count'])
        if require_all_chilren:
            is_complete = data['count'] == len(md_child_ids)
            data = data.loc[is_complete]
        data = data.loc[:, 'sum'].rename(agg_var).reset_index()
        data['location_id'] = parent_id

        if draw_level:
            data = data.loc[:, ['draw', 'location_id', 'date', agg_var]]
        else:
            data = data.loc[:, ['location_id', 'date', agg_var]]

        return data

from typing import List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import (
    Durations,
)


def enforce_epi_threshold(epi_measures: pd.DataFrame, measure: str, mortality_scalar: pd.Series) -> pd.DataFrame:
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


def format_epi_measures(epi_measures: pd.DataFrame,
                        mr_hierarchy: pd.DataFrame,
                        pred_hierarchy: pd.DataFrame,
                        mortality_scalars: pd.Series,
                        durations: Durations) -> pd.DataFrame:
    deaths = (epi_measures['cumulative_deaths'] * mortality_scalars).rename('cumulative_deaths').dropna()
    deaths = _format_measure(deaths, mr_hierarchy, pred_hierarchy)
    cases = _format_measure(epi_measures['cumulative_cases'], mr_hierarchy, pred_hierarchy)
    admissions = _format_measure(epi_measures['cumulative_hospitalizations'], mr_hierarchy, pred_hierarchy)
    epi_measures = pd.concat([deaths, cases, admissions], axis=1).reset_index()
    max_duration = durations.exposure_to_death

    locations = epi_measures.location_id.unique()
    dates = epi_measures.date
    global_date_range = pd.date_range(dates.min() - pd.Timedelta(days=max_duration), dates.max())
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


def aggregate_data_from_md(data: pd.DataFrame, hierarchy: pd.DataFrame, agg_var: str) -> pd.DataFrame:
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
                             parent_id, md_child_ids, agg_var)
                   for parent_id, md_child_ids in parent_children_pairs]
    data = pd.concat([md_data] + parent_data)

    return data.reset_index(drop=True)


def aggregate(data: pd.DataFrame, parent_id: int, md_child_ids: List[int], agg_var: str) -> pd.DataFrame:
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
        is_complete = data['count'] == len(md_child_ids)
        data = data.loc[is_complete, 'sum'].rename(agg_var).reset_index()
        data['location_id'] = parent_id

        if draw_level:
            data = data.loc[:, ['draw', 'location_id', 'date', agg_var]]
        else:
            data = data.loc[:, ['location_id', 'date', agg_var]]

        return data

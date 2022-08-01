from typing import Dict, List
import pandas as pd

from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)


def preprocess_antivirals(scenario_specifications: Dict,
                          data_interface: PreprocessingDataInterface,
                          parameters: Dict) -> pd.DataFrame:
    hierarchy = data_interface.load_hierarchy('pred')
    hierarchy = hierarchy.loc[hierarchy['most_detailed'] == 1]
    full_date_range = pd.date_range('2019-11-01', '2023-12-31')
    index = pd.MultiIndex.from_product([hierarchy['location_id'].to_list(),
                                        full_date_range],
                               names=['location_id', 'date'])

    data = build_antiviral_coverage(index, hierarchy, scenario_specifications)

    return data


def build_antiviral_coverage(index: pd.Index,
                             hierarchy: pd.DataFrame,
                             scenario_specifications: Dict) -> pd.Series:
    coverage = []
    for spec_name, parameters in scenario_specifications.items():
        location_ids = hierarchy.loc[hierarchy['path_to_top_parent']
                                     .apply(lambda x: match_to_parents(parameters['parent_location_ids'], x)),
                                     'location_id'].to_list()
        _coverage = pd.concat(
            [coverage_scaleup(index, parameters['lr_coverage'], *parameters['scaleup_dates'])
             .rename('antiviral_coverage_lr')
             .loc[location_ids],
             coverage_scaleup(index, parameters['hr_coverage'], *parameters['scaleup_dates'])
             .rename('antiviral_coverage_hr')
             .loc[location_ids],],
            axis=1
        )
        coverage.append(_coverage)
    coverage = pd.concat(coverage).sort_index()

    coverage = coverage.reindex(index).fillna(0)

    return coverage


def match_to_parents(parent_location_ids: List[int], path_to_top_parent: str):
    path_to_top_parent = [int(location_id) for location_id in path_to_top_parent.split(',')]
    overlap = set(parent_location_ids).intersection(path_to_top_parent)

    return len(overlap) > 0


def coverage_scaleup(index: pd.Index,
                     max_coverage: float,
                     date_start: str,
                     date_end: str,) -> pd.Series:
    date_start = pd.Timestamp(date_start)
    date_end = pd.Timestamp(date_end)
    dates = pd.date_range(date_start, date_end)
    coverage = (dates - date_start).days / (date_end - date_start).days
    coverage *= max_coverage

    coverage = (pd.Series(coverage, index=dates)
                .reindex(index, level='date')
                .groupby('location_id')
                .ffill()
                .groupby('location_id')
                .bfill()
                .rename('antiviral_coverage'))

    return coverage

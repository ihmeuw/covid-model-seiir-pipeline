import pandas as pd

from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)


def preprocess_antivirals(data_interface: PreprocessingDataInterface, scenario: str):
    data = build_antiviral_coverage(data_interface, scenario)

    return data


def build_antiviral_coverage(data_interface: PreprocessingDataInterface,
                             scenario: str,
                             max_coverage: float = 0.5):
    hierarchy = data_interface.load_hierarchy('pred')
    hierarchy = hierarchy.loc[hierarchy['most_detailed'] == 1]
    full_date_range = pd.date_range('2019-11-01', '2023-12-31')
    index = pd.MultiIndex.from_product([hierarchy['location_id'].to_list(),
                                        full_date_range],
                               names=['location_id', 'date'])

    high_income = hierarchy.loc[hierarchy['path_to_top_parent'].str.contains(',64,'), 
                                'location_id'].to_list()
    low_middle_income = hierarchy.loc[~hierarchy['path_to_top_parent'].str.contains(',64,'), 
                                'location_id'].to_list()

    coverage = []
    for location_ids, date_start, date_end in [(high_income, '2022-02-01', '2022-05-01'),
                                               (low_middle_income, '2022-08-15', '2022-11-15')]:
        _coverage = coverage_scaleup(index, date_start, date_end, max_coverage)
        coverage.append(_coverage.loc[location_ids])
    coverage = pd.concat(coverage).sort_index()

    if scenario == 'reference':
        coverage.loc[low_middle_income] = 0
    elif scenario != 'global_antivirals':
        raise ValueError(f'Invalid antiviral scenario: {scenario}')

    return coverage


def coverage_scaleup(index: pd.Index, date_start: str, date_end: str, max_coverage: float):
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
                .bfill())

    return coverage

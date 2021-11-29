from typing import Dict, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model import (
    helpers,
)

logger = cli_tools.task_performance_logger


def preprocess_epi_data(data_interface: PreprocessingDataInterface) -> None:
    logger.info('Loading epi data.', context='read')
    mr_hierarchy = data_interface.load_hierarchy('mr').reset_index()
    pred_hierarchy = data_interface.load_hierarchy('pred').reset_index()

    age_pattern_data = data_interface.load_age_pattern_data()
    total_covid_scalars = data_interface.load_raw_total_covid_scalars()
    epi_data = data_interface.load_epi_measures()

    logger.info('Processing epi data.', context='transform')
    age_pattern_data = _process_age_pattern_data(age_pattern_data, pred_hierarchy)
    total_covid_scalars = _process_scalars(total_covid_scalars, pred_hierarchy)
    epi_data = pd.concat([_process_epi_data(data, measure, mr_hierarchy, pred_hierarchy)
                          for measure, data in epi_data.items()], axis=1)

    logger.info('Writing epi data.', context='write')
    data_interface.save_age_patterns(age_pattern_data)
    data_interface.save_total_covid_scalars(total_covid_scalars)
    data_interface.save_reported_epi_data(epi_data)


def _process_age_pattern_data(data: pd.DataFrame, hierarchy: pd.DataFrame):
    # Broadcast over location id.
    data['key'] = 1
    hierarchy = hierarchy.reset_index()
    hierarchy['key'] = 1

    data = (hierarchy
            .loc[:, ['location_id', 'key']]
            .merge(data)
            .set_index(['location_id', 'age_group_years_start', 'age_group_years_end'])
            .sort_index()
            .drop(columns='key'))
    return data


def _process_scalars(data: pd.DataFrame, hierarchy: pd.DataFrame):
    missing_locations = list(set(hierarchy.location_id).difference(data.index))
    if missing_locations:
        logger.warning(f"Missing scalars for the following locations: {missing_locations}.  Filling with 1.")
    data = data.reindex(hierarchy.index, fill_value=1.)
    return data


def _process_epi_data(data: pd.Series, measure: str,
                      mr_hierarchy: pd.DataFrame, pred_hierarchy: pd.DataFrame) -> pd.DataFrame:
    data = (data
            .reset_index()
            .groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, [f'cumulative_{measure}']))
            .reset_index(drop=True))
    data, manipulation_metadata = evil_doings(data, mr_hierarchy, measure)

    extra_locations = set(pred_hierarchy.loc[pred_hierarchy['most_detailed'] == 1, 'location_id'])
    extra_locations = list(extra_locations.difference(mr_hierarchy['location_id']))
    extra_data = data.loc[data['location_id'].isin(extra_locations)].reset_index(drop=True)

    data = helpers.aggregate_data_from_md(data, mr_hierarchy, f'cumulative_{measure}')
    data = (data
            .append(extra_data.loc[:, data.columns])
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))

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


def evil_doings(data: pd.DataFrame, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.DataFrame, Dict]:
    manipulation_metadata = {}
    if input_measure == 'cases':
        pass

    elif input_measure == 'hospitalizations':
        # ## hosp/IHR == admissions too low
        # is_argentina = data['location_id'] == 97
        # data = data.loc[~is_argentina].reset_index(drop=True)
        # manipulation_metadata['argentina'] = 'dropped all hospitalizations'

        ## late, not cumulative on first day
        is_ndl = data['location_id'] == 89
        data = data.loc[~is_ndl].reset_index(drop=True)
        manipulation_metadata['netherlands'] = 'dropped all hospitalizations'

        ## is just march-june 2020
        is_vietnam = data['location_id'] == 20
        data = data.loc[~is_vietnam].reset_index(drop=True)
        manipulation_metadata['vietnam'] = 'dropped all hospitalizations'

        ## is just march-june 2020
        is_murcia = data['location_id'] == 60366
        data = data.loc[~is_murcia].reset_index(drop=True)
        manipulation_metadata['murcia'] = 'dropped all hospitalizations'

        ## partial time series
        pakistan_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '165' in x.split(',')),
                                              'location_id'].to_list()
        is_pakistan = data['location_id'].isin(pakistan_location_ids)
        data = data.loc[~is_pakistan].reset_index(drop=True)
        manipulation_metadata['pakistan'] = 'dropped all hospitalizations'

        ## ECDC is garbage
        ecdc_location_ids = [77, 82, 83, 59, 60, 88, 91, 52, 55]
        is_ecdc = data['location_id'].isin(ecdc_location_ids)
        data = data.loc[~is_ecdc].reset_index(drop=True)
        manipulation_metadata['ecdc_countries'] = 'dropped all hospitalizations'

        ## CLOSE, but seems a little low... check w/ new data
        is_goa = data['location_id'] == 4850
        data = data.loc[~is_goa].reset_index(drop=True)
        manipulation_metadata['goa'] = 'dropped all hospitalizations'

        ## too late, starts March 2021
        is_haiti = data['location_id'] == 114
        data = data.loc[~is_haiti].reset_index(drop=True)
        manipulation_metadata['haiti'] = 'dropped all hospitalizations'

        ## late, starts Jan/Feb 2021 (and is a little low, should check w/ new data)
        is_jordan = data['location_id'] == 144
        data = data.loc[~is_jordan].reset_index(drop=True)
        manipulation_metadata['jordan'] = 'dropped all hospitalizations'

        ## too low then too high? odd series
        is_andorra = data['location_id'] == 74
        data = data.loc[~is_andorra].reset_index(drop=True)
        manipulation_metadata['andorra'] = 'dropped all hospitalizations'

    elif input_measure == 'deaths':
        pass

    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')

    return data, manipulation_metadata

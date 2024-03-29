from typing import Dict, Tuple

import pandas as pd
import numpy as np

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
    total_draws = data_interface.get_n_total_draws()

    age_pattern_data = data_interface.load_age_pattern_data()
    total_covid_scalars = data_interface.load_raw_total_covid_scalars()
    epi_data = data_interface.load_epi_measures()

    logger.info('Processing epi data.', context='transform')
    age_pattern_data = _process_age_pattern_data(age_pattern_data, pred_hierarchy)
    total_covid_scalars = _process_scalars(total_covid_scalars, pred_hierarchy, total_draws)
    epi_data = pd.concat([_process_epi_data(data, measure, mr_hierarchy)
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


def _process_scalars(data: pd.DataFrame, hierarchy: pd.DataFrame, total_draws: int):
    missing_locations = list(set(hierarchy.location_id).difference(data.index))
    if missing_locations:
        logger.warning(f"Missing scalars for the following locations: {missing_locations}. Filling with nan.")
    data = data.reindex(hierarchy.location_id)
    dates = pd.date_range(pd.Timestamp('2019-11-01'), pd.Timestamp('2024-01-01'))
    full_idx = pd.MultiIndex.from_product((hierarchy.location_id, dates), names=('location_id', 'date'))
    data = data.reindex(full_idx, level='location_id')

    num_original_draws = len(data.columns)
    for oversample_draw in range(num_original_draws, total_draws):
        # Draw names are 0-indexed, so this just works out okay.
        data[f'draw_{oversample_draw}'] = data[f'draw_{oversample_draw % num_original_draws}']

    return data


def _process_epi_data(data: pd.Series, measure: str,
                      mr_hierarchy: pd.DataFrame) -> pd.DataFrame:
    data = (data
            .reset_index()
            .groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, [f'cumulative_{measure}']))
            .reset_index(drop=True))
    data, manipulation_metadata = evil_doings(data, mr_hierarchy, measure)
    data = data.set_index(['location_id', 'date']).sort_index()

    return data


def evil_doings(data: pd.DataFrame, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.DataFrame, Dict]:
    manipulation_metadata = {}

    if input_measure in ['cases', 'hospitalizations', 'deaths']:
        drop_all = {
            # Non-public locs with bad performance
            7: 'north_korea',
            23: 'kiribati',
            24: 'marshall_islands',
            25: 'micronesia',
            27: 'samoa',
            28: 'solomon_islands',
            29: 'tonga',
            30: 'vanuatu',
            40: 'turkmenistan',
            66: 'brunei_darussalam',
            176: 'comoros',
            186: 'seychelles',
            298: 'american_samoa',
            320: 'cook_islands',
            349: 'greenland',
            369: 'nauru',
            374: 'niue',
            376: 'northern_mariana_islands',
            380: 'palua',
            413: 'tokelau',
            416: 'tuvalu',
            43867: 'prince_edward_island',
            # Just terrible data
            39: 'tajikistan',
            131: 'nicaragua',
            # 133: 'venezuela',
            183: 'mauritius',
            215: 'sao_tome_and_principe',
            175: 'burundi',
            189: 'tanzania',
            # Stopped reporting
            177: 'djibouti',
        }

        is_in_droplist = data['location_id'].isin(drop_all)
        data = data.loc[~is_in_droplist].reset_index(drop=True)
        for location in drop_all.values():
            manipulation_metadata[location] = 'dropped all data'

    if input_measure == 'cases':
        drop_list = {
        }

        for location_id, location_name in drop_list.items():
            is_location = data['location_id'] == location_id
            data = data.loc[~is_location].reset_index(drop=True)
            manipulation_metadata[location_name] = 'dropped all cases'
        
        dated_drop_list = [
            ## (location_id, location_label, drop_start_date) ##
            (43, 'albania', '2022-12-01'),
            (57, 'belarus', '2022-07-01'),
            (115, 'jamaica', '2022-12-01'),
            (4856, 'karnataka', '2022-11-15'),
            (4870, 'tamil_nadu', '2022-10-15'),
        ]

        for location_id, location_name, drop_start_date in dated_drop_list:
            is_location = data['location_id'] == location_id
            is_drop_start_date_on = data['date'] >= drop_start_date
            data = data.loc[~(is_location & is_drop_start_date_on)].reset_index(drop=True)
            manipulation_metadata[location_name] = f'dropped cases beginning {drop_start_date}'

    elif input_measure == 'hospitalizations':
        drop_list = {
            89: 'netherlands',  # late, not cumulative on first day
            20: 'vietnam',  # is just march-june 2020
            60366: 'murcia',  # is just march-june 2020
            114: 'haiti',  # too late, starts March 2021
            74: 'andorra',  # too low then too high? odd series
            209: 'guinea_bissau',  # late, starts in Feb 2021 (also probably too low)
            198: 'zimbabwe',  # late, starts in June 2021 (also too low)
            182: 'malawi',  # too low
            # Incomplete
            179: 'ethiopia',
            184: 'mozambique',
            191: 'zambia',
            60360: 'castilla_la_mancha',
            60361: 'community_of_madrid',
            80: 'france',
            4636: 'wales',
            # Check with new data
            144: 'jordan',  # late, starts Jan/Feb 2021
            4850: 'goa',
        }

        for location_id, location_name in drop_list.items():
            is_location = data['location_id'] == location_id
            data = data.loc[~is_location].reset_index(drop=True)
            manipulation_metadata[location_name] = 'dropped all hospitalizations'

        dated_drop_list = [
            ## (location_id, location_label, drop_start_date) ##
            (422, 'us_virgin_islands', '2022-11-22'),
            (126, 'costa_rica', '2022-04-01'),
        ]

        for location_id, location_name, drop_start_date in dated_drop_list:
            is_location = data['location_id'] == location_id
            is_drop_start_date_on = data['date'] >= drop_start_date
            data = data.loc[~(is_location & is_drop_start_date_on)].reset_index(drop=True)
            manipulation_metadata[location_name] = f'dropped deaths beginning {drop_start_date}'

        ## partial time series
        pakistan_location_ids = hierarchy.loc[
            hierarchy['path_to_top_parent'].apply(lambda x: '165' in x.split(',')),
            'location_id'].to_list()
        is_pakistan = data['location_id'].isin(pakistan_location_ids)
        data = data.loc[~is_pakistan].reset_index(drop=True)
        manipulation_metadata['pakistan'] = 'dropped all hospitalizations'

        ## ECDC is garbage
        ecdc_location_ids = [77, 82, 83, 59, 60, 88, 91, 52, 55]
        is_ecdc = data['location_id'].isin(ecdc_location_ids)
        data = data.loc[~is_ecdc].reset_index(drop=True)
        manipulation_metadata['ecdc_countries'] = 'dropped all hospitalizations'

    elif input_measure == 'deaths':
        drop_list = {
        }

        for location_id, location_name in drop_list.items():
            is_location = data['location_id'] == location_id
            data = data.loc[~is_location].reset_index(drop=True)
            manipulation_metadata[location_name] = 'dropped all deaths'

        dated_drop_list = [
            ## (location_id, location_label, drop_start_date) ##
            (37, 'kyrgyzstan', '2022-03-15'),
            (41, 'uzbekistan', '2022-03-01'),
            (57, 'belarus', '2022-07-01'),
            (109, 'cuba', '2022-04-01'),
            (115, 'jamaica', '2022-11-15'),
            (422, 'us_virgin_islands', '2022-11-22'),
            (150, 'oman', '2022-04-01'),
            (4841, 'andhra_pradesh', '2022-03-01'),
            (4842, 'arunachal_pradesh', '2022-03-01'),
            (4856, 'karnataka', '2022-12-01'),
            (4871, 'telangana', '2022-03-01'),
            (53615, 'azad_jammu_&_kashmir', '2022-04-01'),
            (53616, 'balochistan', '2022-04-01'),
            (53617, 'gilgit_baltistan', '2022-04-01'),
            (10, 'cambodia', '2022-04-01'),
            (170, 'congo', '2022-04-01'),
            (172, 'equatorial_guinea', '2022-02-14'),
            (173, 'gabon', '2022-04-01'),
            (178, 'eritrea', '2022-03-01'),
            (187, 'somalia', '2022-04-01'),
            (435, 'south_sudan', '2022-04-01'),
            (200, 'benin', '2022-04-01'),
            (201, 'burkina_faso', '2022-04-01'),
            (210, 'liberia', '2022-04-01'),
            (212, 'mauritania', '2022-10-15'),
        ]

        for location_id, location_name, drop_start_date in dated_drop_list:
            is_location = data['location_id'] == location_id
            is_drop_start_date_on = data['date'] >= drop_start_date
            data = data.loc[~(is_location & is_drop_start_date_on)].reset_index(drop=True)
            manipulation_metadata[location_name] = f'dropped deaths beginning {drop_start_date}'

    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')

    return data, manipulation_metadata


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
    epi_data = terminal_date_alignment(epi_data)

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


def terminal_date_alignment(data: pd.DataFrame, cutoff: int = 3):
    measures = ['cumulative_cases', 'cumulative_hospitalizations']
    _data = data.loc[:, measures]
    _data = _data.loc[_data.notnull().any(axis=1)]
    terminal_dates = (_data
                      .reset_index()
                      .groupby('location_id')['date'].max()
                      .rename('terminal_date'))
    del _data
    tail_missingness = pd.concat([
        (terminal_dates - data.loc[:, measure].dropna().reset_index().groupby('location_id')['date'].max()).dt.days
        for measure in measures
    ], axis=1)
    tail_missingness = tail_missingness.replace(0, np.nan).min(axis=1).rename('tail_missingness')
    longer_window_locs = [90, 4643, 4657, 4669, 60371]
    trim_idx = tail_missingness <= cutoff
    trim_idx.loc[longer_window_locs] = True
    tail_missingness = tail_missingness.loc[trim_idx]
    terminal_dates = pd.concat([terminal_dates, tail_missingness], axis=1)
    terminal_dates['tail_missingness'] = terminal_dates['tail_missingness'].fillna(0).astype(int)

    accounting = terminal_dates.loc[terminal_dates['tail_missingness'] > 0]
    n_loc_days = accounting['tail_missingness'].sum()
    locs = [str(loc) for loc in accounting.index]
    n_locs = len(locs)
    logger.warning(f'Dropping {n_loc_days} days of data across {n_locs} locations to align terminal '
                   f'dates for case and admissions data when difference is {cutoff} days or less. '
                   f"Affected locations:\n{accounting['tail_missingness'].sort_values(ascending=False)}")

    terminal_dates = terminal_dates.apply(
        lambda x: x['terminal_date'] - pd.Timedelta(days=x['tail_missingness']),
        axis=1
    ).rename('terminal_date')
    data = data.join(terminal_dates, how='left')
    if data['terminal_date'].isnull().any():
        raise ValueError('Mismatch in terminal date aligment.')

    data = data.reset_index('date')
    data = data.loc[data['date'] <= data['terminal_date']]
    del data['terminal_date']
    data = data.set_index('date', append=True)

    return data


def evil_doings(data: pd.DataFrame, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.DataFrame, Dict]:
    manipulation_metadata = {}

    if input_measure in ['cases', 'hospitalizations', 'deaths']:
        drop_all = {
            176: 'comoros',
            172: 'equatorial guinea',
            183: 'mauritius',
            349: 'greenland',
            23: 'kiribati',
        }
        is_in_droplist = data['location_id'].isin(drop_all)
        data = data.loc[~is_in_droplist].reset_index(drop=True)
        for location in drop_all.values():
            manipulation_metadata[location] = 'dropped all data'

    if input_measure == 'cases':
        pass

    elif input_measure == 'hospitalizations':
        ## ## ## ## ## ## ## ## ## ## ## ## ## ##
        ## N Ireland - compositional bias problem; lose last few days of hosp
        is_n_ireland = data['location_id'] == 433
        n_ireland_last_day = data.loc[is_n_ireland, 'date'].max()
        is_n_ireland_sub5 = data['date'] <= n_ireland_last_day - pd.Timedelta(days=5)
        data = data.loc[~is_n_ireland | is_n_ireland_sub5].reset_index(drop=True)
        manipulation_metadata['n_ireland'] = 'dropped all hospitalizations'
        ## ## ## ## ## ## ## ## ## ## ## ## ## ##

        ## hosp/IHR == admissions too low
        is_argentina = data['location_id'] == 97
        data = data.loc[~is_argentina].reset_index(drop=True)
        manipulation_metadata['argentina'] = 'dropped all hospitalizations'

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
        
        ## is just july 2020-april 2021
        is_la_rioja = data['location_id'] == 60376
        data = data.loc[~is_la_rioja].reset_index(drop=True)
        manipulation_metadata['la_rioja'] = 'dropped all hospitalizations'

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
        
        ## late, starts in Feb 2021 (also probably too low)
        is_guinea_bissau = data['location_id'] == 209
        data = data.loc[~is_guinea_bissau].reset_index(drop=True)
        manipulation_metadata['guinea_bissau'] = 'dropped all hospitalizations'
        
        ## late, starts in June 2021 (also too low)
        is_zimbabwe = data['location_id'] == 198
        data = data.loc[~is_zimbabwe].reset_index(drop=True)
        manipulation_metadata['zimbabwe'] = 'dropped all hospitalizations'
        
        ## too low
        is_malawi = data['location_id'] == 182
        data = data.loc[~is_malawi].reset_index(drop=True)
        manipulation_metadata['malawi'] = 'dropped all hospitalizations'

    elif input_measure == 'deaths':
        ## Virginia deaths spike
        is_virginia = data['location_id'] == 569
        is_feb = data['date'] >= pd.Timestamp('2022-02-01')
        data = data.loc[~(is_virginia & is_feb)].reset_index(drop=True)
        manipulation_metadata['virginia'] = 'dropped deaths Feb onward (TEMPORARY)'
        # pass

    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')

    return data, manipulation_metadata

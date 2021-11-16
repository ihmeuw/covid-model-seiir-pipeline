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
    age_pattern_data = data_interface.load_age_pattern_data()
    total_covid_scalars = data_interface.load_raw_total_covid_scalars()
    global_serology = data_interface.load_raw_serology_data()
    cases, hospitalizations, deaths = data_interface.load_epi_measures()

    logger.info('Processing epi data.', context='transform')
    global_serology = _process_serology_data(global_serology)

    logger.info('Writing epi data.', context='write')
    data_interface.save_age_patterns(age_pattern_data)
    data_interface.save_total_covid_scalars(total_covid_scalars)
    data_interface.save_global_serology(global_serology)


def _process_serology_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.info(f'Initial observation count: {len(data)}')

    # date formatting
    if 'Date' in data.columns:
        if 'date' in data.columns:
            raise ValueError('Both `Date` and `date` in serology data.')
        else:
            data = data.rename(columns={'Date': 'date'})
    for date_var in ['start_date', 'date']:
        # data[date_var] = helpers.str_fmt(data[date_var]).replace('.202$', '.2020')
        # data.loc[(data['location_id'] == 570) & (data[date_var] == '11.08.2021'), date_var] = '11.08.2020'
        # data.loc[(data['location_id'] == 533) & (data[date_var] == '13.11.2.2020'), date_var] = '13.11.2020'
        # data.loc[data[date_var] == '05.21.2020', date_var] = '21.05.2020'
        data[date_var] = pd.to_datetime(data[date_var])  # , format='%d.%m.%Y'

    # if no start date provided, assume 2 weeks before end date?
    data['start_date'] = data['start_date'].fillna(data['date'] - pd.Timedelta(days=14))

    # # use mid-point instead of end date
    # data = data.rename(columns={'date':'end_date'})
    # data['n_midpoint_days'] = (data['end_date'] - data['start_date']).dt.days / 2
    # data['n_midpoint_days'] = data['n_midpoint_days'].astype(int)
    # data['date'] = data.apply(lambda x: x['end_date'] - pd.Timedelta(days=x['n_midpoint_days']), axis=1)
    # del data['n_midpoint_days']

    # convert to m/l/u to 0-1, sample size to numeric
    if not (helpers.str_fmt(data['units']).unique() == 'percentage').all():
        raise ValueError('Units other than percentage present.')

    for value_var in ['value', 'lower', 'upper']:
        if data[value_var].dtype.name == 'object':
            data[value_var] = helpers.str_fmt(data[value_var]).replace('not specified', np.nan).astype(float)
        if data[value_var].dtype.name != 'float64':
            raise ValueError(f'Unexpected type for {value_var} column.')

    data['seroprevalence'] = data['value'] / 100
    data['seroprevalence_lower'] = data['lower'] / 100
    data['seroprevalence_upper'] = data['upper'] / 100
    data['sample_size'] = (helpers.str_fmt(data['sample_size'])
                           .replace(('unchecked', 'not specified'), np.nan).astype(float))

    data['bias'] = (helpers.str_fmt(data['bias'])
                    .replace(('unchecked', 'not specified'), '0')
                    .fillna('0')
                    .astype(int))

    data['manufacturer_correction'] = (helpers.str_fmt(data['manufacturer_correction'])
                                       .replace(('not specified', 'not specifed'), '0')
                                       .fillna('0')
                                       .astype(int))

    data['test_target'] = helpers.str_fmt(data['test_target']).str.lower()

    data['study_start_age'] = (helpers.str_fmt(data['study_start_age'])
                               .replace(('not specified'), np.nan).astype(float))
    data['study_end_age'] = (helpers.str_fmt(data['study_end_age'])
                             .replace(('not specified'), np.nan).astype(float))

    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## manually specify certain tests when reporting is mixed (might just be US?)
    # Oxford "mixed" is spike
    is_oxford = data['test_name'] == 'University of Oxford ELISA IgG'
    is_mixed = data['test_target'] == 'mixed'
    data.loc[is_oxford & is_mixed, 'test_target'] = 'spike'

    # Peru N-Roche has the wrong isotype
    is_peru = data['location_id'] == 123
    is_roche = data['test_name'] == 'Roche Elecsys N pan-Ig'
    data.loc[is_peru & is_roche, 'isotype'] = 'pan-Ig'

    # New York (after Nov 2020 onwards, nucleocapsid test is Abbott, not Roche)
    # ADDENDUM (2021-08-31): mixed portion looks the same as the Abbott; recode that as well
    is_ny = data['location_id'] == 555
    is_cdc = data['survey_series'] == 'cdc_series'
    # is_N = data['test_target'] == 'nucleocapsid'
    is_nov_or_later = data['date'] >= pd.Timestamp('2020-11-01')
    data.loc[is_ny & is_cdc & is_nov_or_later, 'isotype'] = 'pan-Ig'
    data.loc[is_ny & is_cdc & is_nov_or_later, 'test_target'] = 'nucleocapsid'  # & is_N
    data.loc[is_ny & is_cdc & is_nov_or_later, 'test_name'] = 'Abbott Architect IgG; Roche Elecsys N pan-Ig'  # & is_N

    # Louisiana mixed portion looks the same as the nucleocapsid; recode (will actually use average)
    is_la = data['location_id'] == 541
    is_cdc = data['survey_series'] == 'cdc_series'
    is_nov_or_later = data['date'] >= pd.Timestamp('2020-11-01')
    data.loc[is_la & is_cdc & is_nov_or_later, 'isotype'] = 'pan-Ig'
    data.loc[is_la & is_cdc & is_nov_or_later, 'test_target'] = 'nucleocapsid'
    data.loc[is_la & is_cdc & is_nov_or_later, 'test_name'] = 'Abbott Architect IgG; Roche Elecsys N pan-Ig'

    # BIG CDC CHANGE
    # many states are coded as Abbott, seem be Roche after the changes in Nov; recode
    for location_id in [523,  # Alabama
                        526,  # Arkansas
                        527,  # California
                        530,  # Delaware
                        531,  # District of Columbia
                        532,  # Florida
                        536,  # Illinois
                        540,  # Kentucky
                        545,  # Michigan
                        547,  # Mississippi
                        548,  # Missouri
                        551,  # Nevada
                        556,  # North Carolina
                        558,  # Ohio
                        563,  # South Carolina
                        564,  # South Dakota
                        565,  # Tennessee
                        566,  # Texas
                        567,  # Utah
                        571,  # West Virginia
                        572,  # Wisconsin
                        573,  # Wyoming
                        ]:
        is_state = data['location_id'] == location_id
        is_cdc = data['survey_series'] == 'cdc_series'
        is_N = data['test_target'] == 'nucleocapsid'
        is_nov_or_later = data['date'] >= pd.Timestamp('2020-11-01')
        data.loc[is_state & is_cdc & is_nov_or_later & is_N, 'isotype'] = 'pan-Ig'
        data.loc[is_state & is_cdc & is_nov_or_later & is_N, 'test_target'] = 'nucleocapsid'
        data.loc[
            is_state & is_cdc & is_nov_or_later & is_N, 'test_name'] = 'Roche Elecsys N pan-Ig'  # 'Abbott Architect IgG; Roche Elecsys N pan-Ig'
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    ## un-outlier Nigeria point before looking at that variable
    data.loc[(data['location_id'] == 214) &
             (data['manual_outlier'] == 1) &
             (data['notes'].str.startswith('Average of ncdc_nimr study')),
             'manual_outlier'] = 0

    outliers = []
    data['manual_outlier'] = data['manual_outlier'].astype(float)
    data['manual_outlier'] = data['manual_outlier'].fillna(0)
    data['manual_outlier'] = data['manual_outlier'].astype(int)
    manual_outlier = data['manual_outlier']
    outliers.append(manual_outlier)
    logger.info(f'{manual_outlier.sum()} rows from sero data flagged as outliers in ETL.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
    ## SOME THINGS
    # # 1)
    # #    Question: What if survey is only in adults? Only kids?
    # #    Current approach: Drop beyond some threshold limits.
    # #    Final solution: ...
    # max_start_age = 30
    # min_end_age = 60
    # data['study_start_age'] = helpers.str_fmt(data['study_start_age']).replace('not specified', np.nan).astype(float)
    # data['study_end_age'] = helpers.str_fmt(data['study_end_age']).replace('not specified', np.nan).astype(float)
    # too_old = data['study_start_age'] > max_start_age
    # too_young = data['study_end_age'] < min_end_age
    # age_outlier = (too_old  | too_young).astype(int)
    # outliers.append(age_outlier)
    # if verbose:
    #     logger.info(f'{age_outlier.sum()} rows from sero data do not have enough '
    #             f'age coverage (at least ages {max_start_age} to {min_end_age}).')

    # 2)
    #    Question: Use of geo_accordance?
    #    Current approach: Drop non-represeentative (geo_accordance == 0).
    #    Final solution: ...
    data['geo_accordance'] = helpers.str_fmt(data['geo_accordance']).replace(('unchecked', np.nan), '0').astype(int)
    geo_outlier = data['geo_accordance'] == 0
    outliers.append(geo_outlier)
    logger.info(f'{geo_outlier.sum()} rows from sero data do not have `geo_accordance`.')
    data['correction_status'] = helpers.str_fmt(data['correction_status']).replace(
        ('unchecked', 'not specified', np.nan), '0').astype(int)

    # 3) Level threshold - 3%
    is_sub3 = data['seroprevalence'] < 0.03
    outliers.append(is_sub3)
    logger.info(f'{is_sub3.sum()} rows from sero data dropped due to having values below 3%.')

    # 4) Extra drops
    # vaccine debacle, lose all the UK spike data in 2021
    is_uk = data['location_id'].isin([4749, 433, 434, 4636])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-01-01')

    uk_vax_outlier = is_uk & is_spike & is_2021
    outliers.append(uk_vax_outlier)
    logger.info(f'{uk_vax_outlier.sum()} rows from sero data dropped due to UK vax issues.')

    # vaccine debacle, lose all the Danish data from Feb 2021 onward
    is_den = data['location_id'].isin([78])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-02-01')

    den_vax_outlier = is_den & is_spike & is_2021
    outliers.append(den_vax_outlier)
    logger.info(f'{den_vax_outlier.sum()} rows from sero data dropped due to Denmark vax issues.')

    # vaccine debacle, lose all the Estonia and Netherlands data from Junue 2021 onward
    is_est_ndl = data['location_id'].isin([58, 89])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-06-01')

    est_ndl_vax_outlier = is_est_ndl & is_spike & is_2021
    outliers.append(est_ndl_vax_outlier)
    logger.info(f'{est_ndl_vax_outlier.sum()} rows from sero data dropped due to Netherlands and Estonia vax issues.')

    # vaccine debacle, lose all the Puerto Rico data from April 2021 onward
    is_pr = data['location_id'].isin([385])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-04-01')

    pr_vax_outlier = is_pr & is_spike & is_2021
    outliers.append(pr_vax_outlier)
    logger.info(f'{pr_vax_outlier.sum()} rows from sero data dropped due to Puerto Rico vax issues.')

    # King/Snohomish data is too early
    is_k_s = data['location_id'] == 60886

    outliers.append(is_k_s)
    logger.info(f'{is_k_s.sum()} rows from sero data dropped from to early King/Snohomish data.')
    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    keep_columns = ['data_id', 'nid', 'survey_series', 'location_id', 'start_date', 'date',
                    'seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper', 'sample_size',
                    'study_start_age', 'study_end_age',
                    'test_name', 'test_target', 'isotype',
                    'bias', 'bias_type',
                    'manufacturer_correction', 'geo_accordance',
                    'is_outlier', 'manual_outlier']
    data['is_outlier'] = pd.concat(outliers, axis=1).max(axis=1).astype(int)
    data = (data
            .sort_values(['location_id', 'is_outlier', 'survey_series', 'date'])
            .reset_index(drop=True))
    data['data_id'] = data.index
    data = data.loc[:, keep_columns]

    logger.info(f"Final ETL inlier count: {len(data.loc[data['is_outlier'] == 0])}")
    logger.info(f"Final ETL outlier count: {len(data.loc[data['is_outlier'] == 1])}")

    return data


def _process_deaths()

def reported_epi(model_inputs_root: Path, input_measure: str,
                 hierarchy: pd.DataFrame, gbd_hierarchy: pd.DataFrame,
                 excess_mortality: bool = None,
                 excess_mortality_draw: int = None, ) -> Tuple[pd.Series, pd.Series]:
    if input_measure == 'deaths':
        if type(excess_mortality) != bool:
            raise TypeError('Must specify `excess_mortality` argument to load deaths.')
        data_path = model_inputs_root / 'full_data_unscaled.csv'
    else:
        data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Confirmed': 'cumulative_cases',
                                'Hospitalizations': 'cumulative_hospitalizations',
                                'Deaths': 'cumulative_deaths', })
    data['date'] = pd.to_datetime(data['Date'])
    keep_cols = ['location_id', 'date', f'cumulative_{input_measure}']
    data = data.loc[:, keep_cols].dropna()
    data['location_id'] = data['location_id'].astype(int)
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)

    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, [f'cumulative_{input_measure}']))
            .reset_index(drop=True))

    data, manipulation_metadata = evil_doings(data, hierarchy, input_measure)

    if excess_mortality:
        # NEED TO UPDATE PATH
        em_scalar_data = estimates.excess_mortailty_scalars(excess_mortality)
        if excess_mortality_draw == -1:
            em_scalar_data = em_scalar_data.groupby('location_id', as_index=False)['em_scalar'].mean()
        else:
            em_scalar_data = em_scalar_data.loc[excess_mortality_draw].reset_index(drop=True)

        data = data.merge(em_scalar_data, how='left')
        missing_locations = data.loc[data['em_scalar'].isnull(), 'location_id'].astype(str).unique().tolist()
        missing_covid_locations = [l for l in missing_locations if l in hierarchy['location_id'].to_list()]
        missing_gbd_locations = [l for l in missing_locations if l in gbd_hierarchy['location_id'].to_list()]
        missing_gbd_locations = [l for l in missing_gbd_locations if l not in missing_covid_locations]
        if missing_covid_locations:
            logger.warning(
                f"Missing scalars for the following Covid hierarchy locations: {', '.join(missing_covid_locations)}")
        if missing_gbd_locations:
            logger.warning(
                f"Missing scalars for the following GBD hierarchy locations: {', '.join(missing_gbd_locations)}")
        data['em_scalar'] = data['em_scalar'].fillna(1)
        data['cumulative_deaths'] *= data['em_scalar']
        del data['em_scalar']
    else:
        logger.info('Using unscaled deaths.')

    extra_locations = gbd_hierarchy.loc[gbd_hierarchy['most_detailed'] == 1, 'location_id'].to_list()
    extra_locations = [l for l in extra_locations if l not in hierarchy['location_id'].to_list()]

    extra_data = data.loc[data['location_id'].isin(extra_locations)].reset_index(drop=True)
    data = helpers.aggregate_data_from_md(data, hierarchy, f'cumulative_{input_measure}')
    data = (data
            .append(extra_data.loc[:, data.columns])
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))

    data[f'daily_{input_measure}'] = (data
                                      .groupby('location_id')[f'cumulative_{input_measure}']
                                      .apply(lambda x: x.diff().fillna(x)))
    data = data.dropna()
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())

    cumulative_data = data[f'cumulative_{input_measure}']
    daily_data = data[f'daily_{input_measure}']

    return cumulative_data, daily_data


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

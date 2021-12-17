import multiprocessing

from loguru import logger
import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import math, utilities
from covid_model_seiir_pipeline.pipeline.preprocessing.model import helpers


def process_raw_serology_data(data: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f'Initial observation count: {len(data)}')

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
    logger.debug(f'{manual_outlier.sum()} rows from sero data flagged as outliers in ETL.')
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
    #     logger.debug(f'{age_outlier.sum()} rows from sero data do not have enough '
    #             f'age coverage (at least ages {max_start_age} to {min_end_age}).')

    # 2)
    #    Question: Use of geo_accordance?
    #    Current approach: Drop non-represeentative (geo_accordance == 0).
    #    Final solution: ...
    data['geo_accordance'] = helpers.str_fmt(data['geo_accordance']).replace(('unchecked', np.nan), '0').astype(int)
    geo_outlier = data['geo_accordance'] == 0
    outliers.append(geo_outlier)
    logger.debug(f'{geo_outlier.sum()} rows from sero data do not have `geo_accordance`.')
    data['correction_status'] = helpers.str_fmt(data['correction_status']).replace(
        ('unchecked', 'not specified', np.nan), '0').astype(int)

    # 3) Extra drops
    # vaccine debacle, lose all the UK spike data in 2021
    is_uk = data['location_id'].isin([4749, 433, 434, 4636])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-01-01')

    uk_vax_outlier = is_uk & is_spike & is_2021
    outliers.append(uk_vax_outlier)
    logger.debug(f'{uk_vax_outlier.sum()} rows from sero data dropped due to UK vax issues.')

    # vaccine debacle, lose all the Danish data from Feb 2021 onward
    is_den = data['location_id'].isin([78])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-02-01')

    den_vax_outlier = is_den & is_spike & is_2021
    outliers.append(den_vax_outlier)
    logger.debug(f'{den_vax_outlier.sum()} rows from sero data dropped due to Denmark vax issues.')

    # vaccine debacle, lose all the Estonia and Netherlands data from Junue 2021 onward
    is_est_ndl = data['location_id'].isin([58, 89])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-06-01')

    est_ndl_vax_outlier = is_est_ndl & is_spike & is_2021
    outliers.append(est_ndl_vax_outlier)
    logger.debug(f'{est_ndl_vax_outlier.sum()} rows from sero data dropped due to Netherlands and Estonia vax issues.')

    # vaccine debacle, lose all the Puerto Rico data from April 2021 onward
    is_pr = data['location_id'].isin([385])
    is_spike = data['test_target'] == 'spike'
    is_2021 = data['date'] >= pd.Timestamp('2021-04-01')

    pr_vax_outlier = is_pr & is_spike & is_2021
    outliers.append(pr_vax_outlier)
    logger.debug(f'{pr_vax_outlier.sum()} rows from sero data dropped due to Puerto Rico vax issues.')

    # King/Snohomish data is too early
    is_k_s = data['location_id'] == 60886

    outliers.append(is_k_s)
    logger.debug(f'{is_k_s.sum()} rows from sero data dropped from to early King/Snohomish data.')
    
    # 4) Level threshold - 3%
    data['tmp_outlier'] = pd.concat(outliers, axis=1).max(axis=1).astype(int)
    is_maxsub3 = (data
                  .groupby(['location_id', 'tmp_outlier'])['seroprevalence']
                  .apply(lambda x: x.max() < 0.03)
                  .rename('is_maxsub3')
                  .reset_index())
    is_maxsub3 = data.merge(is_maxsub3, how='left').loc[data.index, 'is_maxsub3']
    del data['tmp_outlier']
    is_sub1 = data['seroprevalence'] < 0.01
    is_maxsub3_sub1 = is_maxsub3 | is_sub1
    outliers.append(is_maxsub3_sub1)
    logger.debug(f'{is_maxsub3_sub1.sum()} rows from sero data dropped due to having values'
                 'below 1% or a location max below 3%.')
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

    logger.debug(f"Final ETL inlier count: {len(data.loc[data['is_outlier'] == 0])}")
    logger.debug(f"Final ETL outlier count: {len(data.loc[data['is_outlier'] == 1])}")

    return data


def assign_assay(seroprevalence: pd.DataFrame, assay_map: pd.DataFrame) -> pd.DataFrame:
    seroprevalence = seroprevalence.merge(assay_map, how='left')
    missing_match = seroprevalence['assay_map'].isnull()
    is_N = seroprevalence['test_target'] == 'nucleocapsid'
    is_S = seroprevalence['test_target'] == 'spike'
    is_other = ~(is_N | is_S)
    seroprevalence.loc[missing_match & is_N, 'assay_map'] = 'N-Roche, N-Abbott'
    seroprevalence.loc[missing_match & is_S, 'assay_map'] = (
        'S-Roche, S-Ortho Ig, S-Ortho IgG, S-DiaSorin, S-EuroImmun, S-Oxford'
    )
    seroprevalence.loc[missing_match & is_other, 'assay_map'] = (
        'N-Roche, N-Abbott, S-Roche, S-Ortho Ig, S-Ortho IgG, S-DiaSorin, S-EuroImmun, S-Oxford'
    )

    inlier_no_assay = (seroprevalence['is_outlier'] == 0) & seroprevalence['assay_map'].isnull()
    if inlier_no_assay.any():
        raise ValueError(f"Unmapped seroprevalence data: {seroprevalence.loc[inlier_no_assay]}")
    return seroprevalence


def sample_seroprevalence(seroprevalence: pd.DataFrame,
                          n_samples: int,
                          correlate_samples: bool,
                          bootstrap_samples: bool,
                          min_samples: int = 10,
                          floor: float = 1e-5,
                          logit_se_cap: float = 1.,
                          num_threads: int = 1,
                          progress_bar: bool = False):
    logit_se_from_ci = lambda x: (math.logit(x['seroprevalence_upper']) - math.logit(x['seroprevalence_lower'])) / 3.92
    logit_se_from_ss = lambda x: np.sqrt((x['seroprevalence'] * (1 - x['seroprevalence'])) / x['sample_size']) / \
                                 (x['seroprevalence'] * (1.0 - x['seroprevalence']))

    series_vars = ['location_id', 'is_outlier', 'survey_series', 'date']
    seroprevalence = seroprevalence.sort_values(series_vars).reset_index(drop=True)

    if n_samples >= min_samples:
        logger.debug(f'Producing {n_samples} seroprevalence samples.')
        if (seroprevalence['seroprevalence'] < seroprevalence['seroprevalence_lower']).any():
            mean_sub_low = seroprevalence['seroprevalence'] < seroprevalence['seroprevalence_lower']
            raise ValueError(f'Mean seroprevalence below lower:\n{seroprevalence[mean_sub_low]}')
        if (seroprevalence['seroprevalence'] > seroprevalence['seroprevalence_upper']).any():
            high_sub_mean = seroprevalence['seroprevalence'] > seroprevalence['seroprevalence_upper']
            raise ValueError(f'Mean seroprevalence above upper:\n{seroprevalence[high_sub_mean]}')

        summary_vars = ['seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper']
        seroprevalence[summary_vars] = seroprevalence[summary_vars].clip(floor, 1 - floor)

        logit_mean = math.logit(seroprevalence['seroprevalence'].copy())
        logit_se = logit_se_from_ci(seroprevalence.copy())
        logit_se = logit_se.fillna(logit_se_from_ss(seroprevalence.copy()))
        logit_se = logit_se.fillna(logit_se_cap)
        logit_se = logit_se.clip(0, logit_se_cap)

        random_state = utilities.get_random_state('sample_survey_error')
        logit_samples = random_state.normal(loc=logit_mean.to_frame().values,
                                            scale=logit_se.to_frame().values,
                                            size=(len(seroprevalence), n_samples), )
        samples = math.expit(logit_samples)

        ## CANNOT DO THIS, MOVES SOME ABOVE 1
        # # re-center around original mean
        # samples *= seroprevalence[['seroprevalence']].values / samples.mean(axis=1, keepdims=True)
        if correlate_samples:
            logger.info('Correlating seroprevalence samples within location.')
            series_data = (seroprevalence[[sv for sv in series_vars if sv not in ['survey_series', 'date']]]
                           .drop_duplicates()
                           .reset_index(drop=True))
            series_data['series'] = series_data.index
            series_data = seroprevalence.merge(series_data).reset_index(drop=True)
            series_idx_list = [series_data.loc[series_data['series'] == series].index.to_list()
                               for series in range(series_data['series'].max() + 1)]
            sorted_samples = []
            for series_idx in series_idx_list:
                series_samples = samples[series_idx, :].copy()
                series_draw_idx = series_samples[0].argsort().argsort()
                series_samples = np.sort(series_samples, axis=1)[:, series_draw_idx]
                sorted_samples.append(series_samples)
            samples = np.vstack(sorted_samples)
            ## THIS SORTS THE WHOLE SET
            # samples = np.sort(samples, axis=1)

        seroprevalence = seroprevalence.drop(
            ['seroprevalence', 'seroprevalence_lower', 'seroprevalence_upper', 'sample_size'],
            axis=1)
        sample_list = []
        for n, sample in enumerate(samples.T):
            _sample = seroprevalence.copy()
            _sample['seroprevalence'] = sample
            _sample['n'] = n
            sample_list.append(_sample.reset_index(drop=True))

    elif n_samples > 1:
        raise ValueError(f'If sampling, need at least {min_samples}.')
    else:
        logger.debug('Just using mean seroprevalence.')

        seroprevalence['seroprevalence'] = seroprevalence['seroprevalence'].clip(floor, 1 - floor)
        seroprevalence = seroprevalence.drop(['seroprevalence_lower', 'seroprevalence_upper', 'sample_size'],
                                             axis=1)
        seroprevalence['n'] = 0
        sample_list = [seroprevalence.reset_index(drop=True)]

    if bootstrap_samples:
        if n_samples < min_samples:
            raise ValueError('Not set up to bootstrap means only.')
        with multiprocessing.Pool(num_threads) as p:
            bootstrap_list = list(tqdm.tqdm(p.imap(bootstrap, sample_list), total=n_samples, disable=not progress_bar))
    else:
        bootstrap_list = sample_list

    return bootstrap_list


def bootstrap(sample: pd.DataFrame, ):
    n = sample['n'].unique().item()
    random_state = utilities.get_random_state(f'bootstrap_{n}')
    rows = random_state.choice(sample.index, size=len(sample), replace=True)
    # rows = np.random.choice(sample.index, size=len(sample), replace=True)
    bootstraped_samples = []
    for row in rows:
        bootstraped_samples.append(sample.loc[[row]])

    return pd.concat(bootstraped_samples).reset_index(drop=True).drop('n', axis=1)


def get_pop_vaccinated(age_spec_population: pd.Series, vaccinated: pd.Series):
    age_spec_population = age_spec_population.reset_index()
    population = []
    for age_start in range(0, 25, 5):
        for age_end in [65, 125]:
            _population = (age_spec_population
                           .loc[(age_spec_population['age_group_years_start'] >= age_start) &
                                (age_spec_population['age_group_years_end'] <= age_end)]
                           .groupby('location_id', as_index=False)['population'].sum())
            _population['age_group_years_start'] = age_start
            _population['age_group_years_end'] = age_end
            population.append(_population)
    population = pd.concat(population)
    vaccinated = vaccinated.reset_index().merge(population)
    is_adult_only = vaccinated['age_group_years_end'] == 65

    vaccinated.loc[is_adult_only, 'vaccinated'] = (
        vaccinated.loc[is_adult_only, ['cumulative_adults_vaccinated', 'cumulative_essential_vaccinated']].sum(axis=1)
        / vaccinated.loc[is_adult_only, 'population']
    )
    vaccinated.loc[~is_adult_only, 'vaccinated'] = (
        vaccinated.loc[~is_adult_only, 'cumulative_all_vaccinated']
        / vaccinated.loc[~is_adult_only, 'population']
    )
    vaccinated = vaccinated.loc[:, ['location_id', 'date', 'age_group_years_start', 'age_group_years_end', 'vaccinated']]

    return vaccinated


def remove_vaccinated(seroprevalence: pd.DataFrame,
                      vaccinated: pd.Series, ) -> pd.DataFrame:
    seroprevalence['age_group_years_start'] = seroprevalence['study_start_age'].fillna(20)
    seroprevalence['age_group_years_start'] = np.round(seroprevalence['age_group_years_start'] / 5) * 5
    seroprevalence.loc[seroprevalence['age_group_years_start'] > 20, 'age_group_years_start'] = 20

    seroprevalence['age_group_years_end'] = seroprevalence['study_end_age'].fillna(125)
    seroprevalence.loc[seroprevalence['age_group_years_end'] <= 65, 'age_group_years_end'] = 65
    seroprevalence.loc[seroprevalence['age_group_years_end'] > 65, 'age_group_years_end'] = 125

    ## start
    # seroprevalence = seroprevalence.rename(columns={'date':'end_date'})
    # seroprevalence = seroprevalence.rename(columns={'start_date':'date'})
    ##
    ## midpoint
    seroprevalence = seroprevalence.rename(columns={'date': 'end_date'})
    seroprevalence['n_midpoint_days'] = (seroprevalence['end_date'] - seroprevalence['start_date']).dt.days / 2
    seroprevalence['n_midpoint_days'] = seroprevalence['n_midpoint_days'].astype(int)
    seroprevalence['date'] = seroprevalence.apply(lambda x: x['end_date'] - pd.Timedelta(days=x['n_midpoint_days']),
                                                  axis=1)
    ##
    ## always
    start_len = len(seroprevalence)
    seroprevalence = seroprevalence.merge(vaccinated, how='left')
    if len(seroprevalence) != start_len:
        raise ValueError('Sero data expanded in vax merge.')
    if seroprevalence.loc[seroprevalence['vaccinated'].isnull(), 'date'].max() >= pd.Timestamp('2020-12-01'):
        raise ValueError('Missing vax after model start (2020-12-01).')
    seroprevalence['vaccinated'] = seroprevalence['vaccinated'].fillna(0)
    ##
    ## start
    # seroprevalence = seroprevalence.rename(columns={'date':'start_date'})
    # seroprevalence = seroprevalence.rename(columns={'end_date':'date'})
    ##
    ## midpoint
    del seroprevalence['date']
    del seroprevalence['n_midpoint_days']
    seroprevalence = seroprevalence.rename(columns={'end_date': 'date'})
    ##

    seroprevalence.loc[seroprevalence['test_target'] != 'spike', 'vaccinated'] = 0

    seroprevalence = seroprevalence.rename(columns={'seroprevalence': 'reported_seroprevalence'})

    seroprevalence['seroprevalence'] = 1 - (1 - seroprevalence['reported_seroprevalence']) / (
                1 - seroprevalence['vaccinated'])

    del seroprevalence['vaccinated']

    return seroprevalence

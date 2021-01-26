from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import static_vars


def compute_reimposition_threshold(deaths, population, reimposition_threshold, max_threshold):
    death_rate = deaths.reset_index(level='date').merge(population, on='location_id')
    death_rate['death_rate'] = death_rate['deaths'] / death_rate['population']
    death_rate = (death_rate[death_rate.observed == 1]
                  .groupby('location_id')
                  .apply(lambda x: x.iloc[-14:])
                  .reset_index(level=0, drop=True))
    days_over_death_rate = ((death_rate.death_rate > reimposition_threshold.reindex(death_rate.index))
                            .groupby('location_id')
                            .sum())
    reimposition_threshold.loc[days_over_death_rate >= 7] = max_threshold / 1e6
    return reimposition_threshold


def compute_reimposition_date(deaths, population, reimposition_threshold,
                              min_wait, last_reimposition_end_date) -> pd.Series:
    death_rate = deaths.reset_index(level='date').merge(population, on='location_id')
    death_rate['death_rate'] = death_rate['deaths'] / death_rate['population']

    projected = death_rate['observed'] == 0
    last_observed_date = death_rate[~projected].groupby('location_id')['date'].max()
    min_reimposition_date = (last_observed_date + min_wait)
    previously_implemented = last_reimposition_end_date[last_reimposition_end_date.notnull()].index
    min_reimposition_date.loc[previously_implemented] = last_reimposition_end_date.loc[previously_implemented] + min_wait

    after_min_reimposition_date = death_rate['date'] >= min_reimposition_date.loc[death_rate.index]
    over_threshold = death_rate['death_rate'] > reimposition_threshold.reindex(death_rate.index)
    reimposition_date = (death_rate[over_threshold & after_min_reimposition_date]
                         .groupby('location_id')['date']
                         .min()
                         .rename('reimposition_date'))

    return reimposition_date


def compute_mobility_lower_bound(mobility: pd.DataFrame, mandate_effect: pd.DataFrame) -> pd.Series:
    min_observed_mobility = mobility.groupby('location_id').min().rename('min_mobility')
    max_mandate_mobility = mandate_effect.sum(axis=1).rename('min_mobility').reindex(min_observed_mobility.index,
                                                                                     fill_value=100)
    mobility_lower_bound = min_observed_mobility.where(min_observed_mobility <= max_mandate_mobility,
                                                       max_mandate_mobility)
    return mobility_lower_bound


def compute_rampup(reimposition_date: pd.Series,
                   percent_mandates: pd.DataFrame,
                   days_on: pd.Timedelta) -> pd.DataFrame:
    rampup = pd.merge(reimposition_date, percent_mandates.reset_index(level='date'), on='location_id', how='left')
    rampup['rampup'] = rampup.groupby('location_id')['percent'].apply(lambda x: x / x.max())
    rampup['first_date'] = rampup.groupby('location_id')['date'].transform('min')
    rampup['diff_date'] = rampup['reimposition_date'] - rampup['first_date']
    rampup['date'] = rampup['date'] + rampup['diff_date'] + days_on
    rampup = rampup.reset_index()[['location_id', 'date', 'rampup']]
    return rampup


def compute_new_mobility(old_mobility: pd.Series,
                         reimposition_date: pd.Series,
                         mobility_lower_bound: pd.Series,
                         percent_mandates: pd.DataFrame,
                         days_on: pd.Timedelta) -> pd.Series:
    mobility = pd.merge(old_mobility.reset_index(level='date'), reimposition_date, how='left', on='location_id')
    mobility = mobility.merge(mobility_lower_bound, how='left', on='location_id')

    reimposes = mobility['reimposition_date'].notnull()
    dates_on = ((mobility['reimposition_date'] <= mobility['date'])
                & (mobility['date'] <= mobility['reimposition_date'] + days_on))
    mobility['mobility_explosion'] = mobility['min_mobility'].where(reimposes & dates_on, np.nan)

    rampup = compute_rampup(reimposition_date, percent_mandates, days_on)

    mobility = mobility.merge(rampup, how='left', on=['location_id', 'date'])
    post_reimplementation = ~(mobility['mobility_explosion'].isnull() & mobility['rampup'].notnull())
    mobility['mobility_explosion'] = mobility['mobility_explosion'].where(
        post_reimplementation,
        mobility['min_mobility'] * mobility['rampup']
    )

    idx_columns = ['location_id', 'date']
    mobility = (mobility[idx_columns + ['mobility', 'mobility_explosion']]
                .set_index(idx_columns)
                .sort_index()
                .min(axis=1))
    return mobility


class MandateReimpositionParams(NamedTuple):
    min_wait: pd.Timedelta
    days_on: pd.Timedelta
    reimposition_threshold: pd.Series
    max_threshold: float


def unpack_parameters(algorithm_parameters: Dict, location_ids: List[int]) -> MandateReimpositionParams:
    min_wait = pd.Timedelta(days=algorithm_parameters['minimum_delay'])
    days_on = pd.Timedelta(days=static_vars.DAYS_PER_WEEK * algorithm_parameters['reimposition_duration'])
    reimposition_threshold = pd.Series(algorithm_parameters['death_threshold'] / 1e6,
                                       index=pd.Index(location_ids, name='location_id'), name='threshold')
    max_threshold = algorithm_parameters['max_threshold']
    return MandateReimpositionParams(min_wait, days_on, reimposition_threshold, max_threshold)

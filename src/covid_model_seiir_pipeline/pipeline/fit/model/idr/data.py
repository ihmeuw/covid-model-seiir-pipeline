import itertools
from typing import Dict, List

from loguru import logger

import pandas as pd
import numpy as np


def create_model_data(cumulative_cases: pd.Series,
                      daily_cases: pd.Series,
                      seroprevalence: pd.DataFrame,
                      testing_capacity: pd.Series,
                      daily_infections: pd.Series,
                      covariates: List,
                      covariate_list: List,
                      durations: Dict,
                      population: pd.Series):
    idr_data = seroprevalence.loc[seroprevalence['is_outlier'] == 0].copy()
    idr_data['date'] -= pd.Timedelta(days=durations['pcr_to_sero'])
    idr_data = (idr_data
                .set_index(['data_id', 'location_id', 'date'])
                .loc[:, 'seroprevalence'])
    idr_data = ((cumulative_cases / (idr_data * population))
                .dropna()
                .rename('idr'))
    
    daily_infections = daily_infections.reset_index()
    
    testing_capacity = testing_capacity.reset_index()
    testing_capacity['date'] -= pd.Timedelta(days=durations['exposure_to_case'])
    sero_location_dates = seroprevalence[['location_id', 'date']].drop_duplicates()
    sero_location_dates = list(zip(sero_location_dates['location_id'], sero_location_dates['date']))
    infwavg_testing_capacity = []
    for location_id, date in sero_location_dates:
        infwavg_testing_capacity.append(
            _get_infection_weighted_avg_testing(
                (daily_infections
                 .loc[(daily_infections['location_id'] == location_id) &
                      (daily_infections['date'] <= (date - pd.Timedelta(days=durations['exposure_to_seroconversion'])))]
                 .set_index(['location_id', 'date'])
                 .loc[:, 'daily_infections']),
                (testing_capacity
                 .loc[(testing_capacity['location_id'] == location_id) &
                      (testing_capacity['date'] <= (date - pd.Timedelta(days=durations['exposure_to_seroconversion'])))]
                 .set_index(['location_id', 'date'])
                 .loc[:, 'testing_capacity']),
            )
        )
    infwavg_testing_capacity = pd.concat(infwavg_testing_capacity,
                                         names='infwavg_testing_capacity').reset_index()
    infwavg_testing_capacity['date'] += pd.Timedelta(days=durations['exposure_to_case'])
    infwavg_testing_capacity = (infwavg_testing_capacity
                                .set_index(['location_id', 'date'])
                                .loc[:, 'infwavg_testing_capacity'])
    daily_infections = daily_infections.set_index(['location_id', 'date'])
    
    log_infwavg_testing_rate_capacity = (np.log(infwavg_testing_capacity / population)
                                         .rename('log_infwavg_testing_rate_capacity'))
    del infwavg_testing_capacity

    # add testing capacity
    model_data = log_infwavg_testing_rate_capacity.to_frame().join(idr_data)
    
    # add covariates
    model_data = model_data.join(pd.concat(covariates, axis=1).loc[:, covariate_list], how='outer')
    
    return model_data.reset_index()


def create_pred_data(hierarchy: pd.DataFrame,
                     population: pd.Series,
                     testing_capacity: pd.Series,
                     covariates: List[pd.Series],
                     covariate_list: List[str],
                     pred_start_date: pd.Timestamp,
                     pred_end_date: pd.Timestamp):
    pred_data = pd.DataFrame(list(itertools.product(hierarchy['location_id'].to_list(),
                                                    list(pd.date_range(pred_start_date, pred_end_date)))),
                             columns=['location_id', 'date'])
    pred_data['intercept'] = 1
    pred_data = pred_data.set_index(['location_id', 'date'])
    log_testing_rate_capacity = np.log(testing_capacity / population).rename('log_testing_rate_capacity')
    pred_data = pred_data.join(log_testing_rate_capacity, how='outer')

    pred_data = pred_data.join(pd.concat(covariates, axis=1).loc[:, covariate_list], how='outer')

    return pred_data.dropna().reset_index()


def _get_infection_weighted_avg_testing(daily_infections: pd.Series,
                                        testing_capacity: pd.Series) -> pd.Series:
    infwavg_data = pd.concat([testing_capacity.rename('testing_capacity'),
                              daily_infections.rename('daily_infections')], axis=1)
    infwavg_data = infwavg_data.loc[infwavg_data['testing_capacity'].notnull()]
    infwavg_data['daily_infections'] = infwavg_data['daily_infections'].fillna(method='bfill')
    if infwavg_data.isnull().any().any():
        logger.warning(f"Missing tail infections for location_id {infwavg_data.reset_index()['location_id'].unique().item()}.")
        infwavg_data['daily_infections'] = infwavg_data['daily_infections'].fillna(method='ffill')
    if not infwavg_data.empty:
        infwavg_testing_capacity = np.average(infwavg_data['testing_capacity'], weights=(infwavg_data['daily_infections'] + 1))
        return pd.Series(infwavg_testing_capacity,
                         name='infwavg_testing_capacity',
                         index=infwavg_data.index[[-1]])
    else:
        return pd.Series(np.array([]),
                         name='infwavg_testing_capacity',
                         index=infwavg_data.index)

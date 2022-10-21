import itertools
from typing import Dict, List

from loguru import logger

import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.rates.date_of_infection import (
    determine_mean_date_of_infection,
)


def create_model_data(cumulative_hospitalizations: pd.Series,
                      daily_hospitalizations: pd.Series,
                      seroprevalence: pd.DataFrame,
                      daily_infections: pd.Series,
                      ratio_data_scalar: pd.Series,
                      covariate_pool: pd.DataFrame,
                      hierarchy: pd.DataFrame,
                      population: pd.Series,
                      day_0: pd.Timestamp,
                      durations: Dict,) -> pd.DataFrame:
    ihr_data = seroprevalence.loc[seroprevalence['is_outlier'] == 0].copy()
    ihr_data['date'] -= pd.Timedelta(days=durations['admission_to_seropositive'])
    ihr_data = (ihr_data
                .set_index(['data_id', 'location_id', 'date'])
                .loc[:, 'seroprevalence'])
    ihr_data = ((cumulative_hospitalizations / (ihr_data * population))
                .dropna()
                .rename('ihr'))

    # add average date of infection
    loc_dates = ihr_data.reset_index()[['location_id', 'date']].drop_duplicates().values.tolist()
    dates_data = determine_mean_date_of_infection(
        location_dates=loc_dates,
        daily_infections=daily_infections.copy(),
        lag=durations['exposure_to_admission'],
    )
    model_data = ihr_data.to_frame().join(dates_data.set_index(['location_id', 'date']), how='left')
    if model_data['mean_infection_date'].isnull().any():
        logger.warning('Missing mean infection date. Dropping data where date is missing')
        model_data = model_data.loc[model_data['mean_infection_date'].notnull()]
    model_data['t'] = (model_data['mean_infection_date'] - day_0).dt.days
    
    # add covariates
    model_data = model_data.join(covariate_pool)

    # add ratio data scalar
    model_data = model_data.join(ratio_data_scalar, how='left')
    if model_data['ratio_data_scalar'].isnull().any():
        raise ValueError('Missing IHR data scalar.')

    return model_data.reset_index()


def create_pred_data(hierarchy: pd.DataFrame,
                     covariate_pool: pd.DataFrame,
                     pred_start_date: pd.Timestamp,
                     pred_end_date: pd.Timestamp,
                     day_0: pd.Timestamp,) -> pd.DataFrame:
    pred_data = pd.DataFrame(list(itertools.product(hierarchy['location_id'].to_list(),
                                                    list(pd.date_range(pred_start_date, pred_end_date)))),
                             columns=['location_id', 'date'])
    pred_data['intercept'] = 1
    pred_data['t'] = (pred_data['date'] - day_0).dt.days
    pred_data = pred_data.set_index(['location_id', 'date'])
    
    pred_data = pred_data.join(covariate_pool, how='left')
    
    return pred_data.dropna().reset_index()

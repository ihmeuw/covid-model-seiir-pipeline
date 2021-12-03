import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model import determine_mean_date_of_infection


def create_model_data(cumulative_hospitalizations: pd.Series,
                      daily_hospitalizations: pd.Series,
                      seroprevalence: pd.DataFrame,
                      daily_infections: pd.Series,
                      variant_prevalence: pd.Series,
                      covariates: List[pd.Series],
                      covariate_list: List[str],
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
    
    # add cumulative variant prevalence
    variant_infections = (variant_prevalence * daily_infections).fillna(0)
    cumulative_variant_prevalence = (
            variant_infections.groupby(level=0).cumsum() /
            daily_infections.groupby(level=0).cumsum()
    ).rename('variant_prevalence')
    cumulative_variant_prevalence = (cumulative_variant_prevalence
                                     .loc[daily_infections.index]
                                     .fillna(0)
                                     .reset_index())
    cumulative_variant_prevalence['date'] += pd.Timedelta(days=durations['exposure_to_admission'])
    cumulative_variant_prevalence = cumulative_variant_prevalence.set_index(['location_id', 'date'])
    model_data = ihr_data.to_frame().join(cumulative_variant_prevalence, how='left')
    if model_data['variant_prevalence'].isnull().any():
        raise ValueError('Missing cumulative variant prevalence.')

    # add average date of infection
    loc_dates = model_data.reset_index()[['location_id', 'date']].drop_duplicates().values.tolist()
    dates_data = determine_mean_date_of_infection(
        location_dates=loc_dates,
        daily_infections=daily_infections.copy(),
        lag=durations['exposure_to_admission'],
    )
    model_data = model_data.join(dates_data.set_index(['location_id', 'date']), how='left')
    if model_data['mean_infection_date'].isnull().any():
        raise ValueError('Missing mean infection date.')
    model_data['t'] = (model_data['mean_infection_date'] - day_0).dt.days
    
    # add covariates
    model_data = model_data.join(pd.concat(covariates, axis=1).loc[:, covariate_list])
            
    return model_data.reset_index()


def create_pred_data(hierarchy: pd.DataFrame,
                     covariates: List[pd.Series],
                     covariate_list: List[str],
                     pred_start_date: pd.Timestamp,
                     pred_end_date: pd.Timestamp,
                     day_0: pd.Timestamp,) -> pd.DataFrame:
    pred_data = pd.DataFrame(list(itertools.product(hierarchy['location_id'].to_list(),
                                                    list(pd.date_range(pred_start_date, pred_end_date)))),
                             columns=['location_id', 'date'])
    pred_data['intercept'] = 1
    pred_data['t'] = (pred_data['date'] - day_0).dt.days
    pred_data['variant_prevalence'] = 0
    pred_data = pred_data.set_index(['location_id', 'date'])
    
    pred_data = pred_data.join(pd.concat(covariates, axis=1).loc[:, covariate_list], how='outer')
    
    return pred_data.dropna().reset_index()

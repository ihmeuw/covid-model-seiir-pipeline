import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


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
    
    variant_infections = (variant_prevalence * daily_infections).fillna(0)
    cumulative_variant_prevalence = (
            variant_infections.groupby(level=0).cumsum() /
            daily_infections.groupby(level=0).cumsum().rename('cumulative_variant_prevalence')
    )
    cumulative_variant_prevalence = (cumulative_variant_prevalence
                                     .loc[daily_infections.index]
                                     .fillna(0)
                                     .reset_index())
    cumulative_variant_prevalence['date'] += pd.Timedelta(days=durations['exposure_to_admission'])
    cumulative_variant_prevalence = cumulative_variant_prevalence.set_index(['location_id', 'date'])

    # get mean day of admission int
    loc_dates = (ihr_data
                 .reset_index()
                 .loc[:, ['location_id', 'date']]
                 .drop_duplicates()
                 .values
                 .tolist())
    time = []
    for location_id, survey_end_date in loc_dates:
        loc_hosps = daily_hospitalizations.loc[location_id]
        loc_hosps = loc_hosps.clip(0, np.inf)
        loc_hosps = loc_hosps.reset_index()
        loc_hosps = loc_hosps.loc[loc_hosps['date'] <= survey_end_date]
        loc_hosps['t'] = (loc_hosps['date'] - day_0).dt.days
        t = np.average(loc_hosps['t'], weights=loc_hosps['daily_hospitalizations'] + 1e-6)
        t = int(np.round(t))
        mean_hospitalization_date = loc_hosps.loc[loc_hosps['t'] == t, 'date'].item()
        
        lt_vp = cumulative_variant_prevalence.loc[(location_id, survey_end_date)].item()
        
        time.append(
            pd.DataFrame(
                {'t':t, 'mean_hospitalization_date':mean_hospitalization_date, 'variant_prevalence': lt_vp,},
                index=pd.MultiIndex.from_arrays([[location_id], [survey_end_date]],
                                                names=('location_id', 'date')),)
        )
    time = pd.concat(time)

    # add time
    model_data = time.join(ihr_data, how='outer')
    
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

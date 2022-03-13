from typing import Dict, List, Tuple

import pandas as pd

from covid_model_seiir_pipeline.lib import math
from covid_model_seiir_pipeline.pipeline.fit.model.rates.idr import (
    data,
    model,
    flooring,
)


def runner(epi_data,
           seroprevalence: pd.DataFrame,
           covariates: List[pd.Series],
           covariate_pool: Dict[str, List[str]],
           mr_hierarchy: pd.DataFrame,
           pred_hierarchy: pd.DataFrame,
           population: pd.Series,
           testing_capacity: pd.Series,
           durations: Dict,
           daily_infections: pd.Series,
           pred_start_date: pd.Timestamp,
           pred_end_date: pd.Timestamp,
           num_threads: int,
           progress_bar: bool,
           **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_cases = epi_data['cumulative_cases'].dropna()
    daily_cases = epi_data['daily_cases'].dropna()
    covariate_list = covariate_pool['idr']

    model_data = data.create_model_data(
        cumulative_cases=cumulative_cases.copy(),
        daily_cases=daily_cases.copy(),
        seroprevalence=seroprevalence.copy(),
        testing_capacity=testing_capacity.copy(),
        daily_infections=daily_infections.copy(),
        covariates=covariates.copy(),
        covariate_list=covariate_list.copy(),
        durations=durations.copy(),
        population=population.copy(),
    )
    pred_data = data.create_pred_data(
        hierarchy=pred_hierarchy.copy(),
        population=population.copy(),
        testing_capacity=testing_capacity.copy(),
        covariates=covariates.copy(),
        covariate_list=covariate_list.copy(),
        pred_start_date=pred_start_date,
        pred_end_date=pred_end_date,
    )
    
    # check what NAs in pred data might be about, get rid of them in safer way
    mr_model_dict, prior_dicts, pred, pred_fe, pred_location_map, level_lambdas = model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        mr_hierarchy=mr_hierarchy.copy(),
        pred_hierarchy=pred_hierarchy.copy(),
        covariate_list=covariate_list.copy(),
        num_threads=num_threads,
        progress_bar=progress_bar,
    )
    
    rmse_data, floor_data = flooring.find_idr_floor(
        pred=pred.copy(),
        daily_cases=daily_cases.copy(),
        serosurveys=(seroprevalence
                     .set_index(['location_id', 'date'])
                     .sort_index()
                     .loc[:, 'seroprevalence']).copy(),
        population=population.copy(),
        hierarchy=pred_hierarchy.copy(),
        test_range=[0.01, 0.1] + list(range(1, 11)),
        num_threads=num_threads,
        progress_bar=progress_bar,
    )
    
    pred = (pred.reset_index().set_index('location_id').join(floor_data, how='left'))
    pred = (pred
            .groupby(level=0)
            .apply(lambda x: math.scale_to_bounds(x.set_index('date', append=True).loc[:, 'pred_idr'],
                                                  x['idr_floor'].unique().item(),
                                                  ceiling=1.,))
            .rename('pred_idr')
            .reset_index(level=0, drop=True)
            .to_frame())

    for col in ['idr', 'idr_lr', 'idr_hr']:
        pred.loc[:, col] = pred['pred_idr']
    pred = pred.drop(columns='pred_idr')
    pred['lag'] = durations["exposure_to_case"]
    
    return pred, model_data
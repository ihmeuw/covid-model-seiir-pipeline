from typing import Dict, List

import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model import age_standardization
from covid_model_seiir_pipeline.pipeline.fit.model.ifr import (
    data,
    model,
)


def runner(cumulative_deaths: pd.Series,
           daily_deaths: pd.Series,
           seroprevalence: pd.DataFrame,
           covariates: List[pd.Series],
           covariate_list: List[str],
           daily_infections: pd.Series,
           variant_prevalence: pd.Series,
           mr_hierarchy: pd.DataFrame,
           pred_hierarchy: pd.DataFrame,
           ifr_age_pattern: pd.Series,
           sero_age_pattern: pd.Series,
           population: pd.Series,
           age_spec_population: pd.Series,
           variant_risk_ratio: float,
           durations: Dict,
           day_inflection: pd.Timestamp,
           day_0: pd.Timestamp,
           pred_start_date: pd.Timestamp,
           pred_end_date: pd.Timestamp,
           num_threads: int,
           progress_bar: bool) -> pd.DataFrame:
    model_data = data.create_model_data(
        cumulative_deaths=cumulative_deaths,
        daily_deaths=daily_deaths,
        seroprevalence=seroprevalence,
        covariates=covariates,
        covariate_list=covariate_list,
        daily_infections=daily_infections,
        variant_prevalence=variant_prevalence,
        hierarchy=mr_hierarchy,
        population=population,
        day_0=day_0,
        durations=durations,
    )
    
    pred_data = data.create_pred_data(
        hierarchy=pred_hierarchy,
        covariates=covariates,
        covariate_list=covariate_list,
        pred_start_date=pred_start_date,
        pred_end_date=pred_end_date,
        day_0=day_0,
    )

    (mr_model_dict, prior_dicts, pred, pred_fe,
     pred_location_map, age_stand_scaling_factor, level_lambdas) = model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        ifr_age_pattern=ifr_age_pattern.copy(),
        sero_age_pattern=sero_age_pattern.copy(),
        age_spec_population=age_spec_population.copy(),
        variant_risk_ratio=variant_risk_ratio,
        mr_hierarchy=mr_hierarchy.copy(),
        pred_hierarchy=pred_hierarchy.copy(),
        day_0=day_0,
        day_inflection=day_inflection,
        covariate_list=covariate_list.copy(),
        num_threads=num_threads,
        progress_bar=progress_bar,
    )
    
    lr_rr, hr_rr = age_standardization.get_risk_group_rr(
        ifr_age_pattern.copy(),
        sero_age_pattern.copy() ** 0,  # just use flat
        age_spec_population.copy(),
    )
    pred_lr = (pred * lr_rr).rename('pred_ifr_lr')
    pred_hr = (pred * hr_rr).rename('pred_ifr_hr')
    
    pred = pd.concat([pred, pred_lr, pred_hr], axis=1)
    
    return pred, model_data

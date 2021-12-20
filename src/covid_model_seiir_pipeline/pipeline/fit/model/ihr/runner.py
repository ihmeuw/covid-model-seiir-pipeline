from typing import Dict, List

import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model import age_standardization
from covid_model_seiir_pipeline.pipeline.fit.model.ihr import (
    data,
    model,
)


def runner(cumulative_hospitalizations: pd.Series,
           daily_hospitalizations: pd.Series,
           seroprevalence: pd.DataFrame,
           covariates: List[pd.Series],
           covariate_list: List[str],
           daily_infections: pd.Series,
           variant_prevalence: pd.Series,
           mr_hierarchy: pd.DataFrame,
           pred_hierarchy: pd.DataFrame,
           ihr_age_pattern: pd.Series,
           sero_age_pattern: pd.Series,
           population: pd.Series,
           age_spec_population: pd.Series,
           variant_risk_ratio: float,
           durations: Dict,
           day_0: pd.Timestamp,
           pred_start_date: pd.Timestamp,
           pred_end_date: pd.Timestamp) -> pd.DataFrame:
    model_data = data.create_model_data(
        cumulative_hospitalizations=cumulative_hospitalizations.copy(),
        daily_hospitalizations=daily_hospitalizations.copy(),
        seroprevalence=seroprevalence.copy(),
        daily_infections=daily_infections.copy(),
        variant_prevalence=variant_prevalence.copy(),
        covariates=covariates.copy(),
        covariate_list=covariate_list.copy(),
        hierarchy=mr_hierarchy.copy(),
        population=population.copy(),
        day_0=day_0,
        durations=durations.copy(),
    )
    pred_data = data.create_pred_data(
        hierarchy=pred_hierarchy.copy(),
        covariates=covariates.copy(),
        covariate_list=covariate_list.copy(),
        pred_start_date=pred_start_date,
        pred_end_date=pred_end_date,
        day_0=day_0,
    )
    # check what NAs in data might be about, get rid of them in safer way
    (mr_model_dict,
     prior_dicts,
     pred,
     pred_fe,
     pred_location_map,
     age_stand_scaling_factor,
     level_lambdas) = model.run_model(
        model_data=model_data.copy(),
        pred_data=pred_data.copy(),
        ihr_age_pattern=ihr_age_pattern.copy(),
        sero_age_pattern=sero_age_pattern.copy(),
        age_spec_population=age_spec_population.copy(),
        mr_hierarchy=mr_hierarchy.copy(),
        pred_hierarchy=pred_hierarchy.copy(),
        covariate_list=covariate_list.copy(),
        variant_risk_ratio=variant_risk_ratio,
    )
    lr_rr, hr_rr = age_standardization.get_risk_group_rr(
        ihr_age_pattern.copy(),
        sero_age_pattern.copy() ** 0,  # just use flat
        age_spec_population.copy(),
    )
    pred_lr = (pred * lr_rr).rename('pred_ihr_lr')
    pred_hr = (pred * hr_rr).rename('pred_ihr_hr')
    
    pred = pd.concat([pred, pred_lr, pred_hr,], axis=1)
    
    return pred, model_data

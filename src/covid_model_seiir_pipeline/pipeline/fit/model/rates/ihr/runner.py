from typing import Dict, List, Tuple

import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.rates import age_standardization
from covid_model_seiir_pipeline.pipeline.fit.model.rates.ihr import (
    data,
    model,
)


def runner(epi_data: pd.DataFrame,
           seroprevalence: pd.DataFrame,
           covariates: List[pd.Series],
           covariate_pool: Dict[str, List[str]],
           daily_infections: pd.Series,
           variant_prevalence: pd.Series,
           mr_hierarchy: pd.DataFrame,
           pred_hierarchy: pd.DataFrame,
           age_patterns: pd.DataFrame,
           population: pd.Series,
           age_specific_population: pd.Series,
           variant_risk_ratio: Dict[str, float],
           durations: Dict,
           day_0: pd.Timestamp,
           pred_start_date: pd.Timestamp,
           pred_end_date: pd.Timestamp,
           num_threads: int,
           progress_bar: bool,
           **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_hospitalizations = epi_data['cumulative_hospitalizations'].dropna()
    daily_hospitalizations = epi_data['daily_hospitalizations'].dropna()
    covariate_list = covariate_pool['ihr']
    ihr_age_pattern = age_patterns['ihr']
    sero_age_pattern = age_patterns['seroprevalence']
    variant_risk_ratio = variant_risk_ratio['ihr']

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
        age_spec_population=age_specific_population.copy(),
        mr_hierarchy=mr_hierarchy.copy(),
        pred_hierarchy=pred_hierarchy.copy(),
        covariate_list=covariate_list.copy(),
        variant_risk_ratio=variant_risk_ratio,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )
    lr_rr, hr_rr = age_standardization.get_risk_group_rr(
        ihr_age_pattern.copy(),
        sero_age_pattern.copy() ** 0,  # just use flat
        age_specific_population.copy(),
    )
    pred_lr = (pred * lr_rr).rename('pred_ihr_lr')
    pred_hr = (pred * hr_rr).rename('pred_ihr_hr')
    
    pred = pd.concat([pred, pred_lr, pred_hr], axis=1)

    pred = pred.rename(columns={
        'pred_ihr_lr': 'ihr_lr', 'pred_ihr_hr': 'ihr_hr', 'pred_ihr': 'ihr'
    })
    pred.loc[:, 'lag'] = durations["exposure_to_admission"]
    
    return pred, model_data

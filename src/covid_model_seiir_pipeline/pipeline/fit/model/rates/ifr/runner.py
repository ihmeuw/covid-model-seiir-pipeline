from typing import Dict, List, Tuple

import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.rates import (
    age_standardization,
    variants,
)
from covid_model_seiir_pipeline.pipeline.fit.model.rates.ifr import (
    data,
    model,
)


def runner(epi_data: pd.DataFrame,
           seroprevalence: pd.DataFrame,
           covariate_pool: pd.DataFrame,
           daily_infections: pd.Series,
           variant_prevalence: pd.DataFrame,
           variant_risk_ratios: pd.DataFrame,
           mr_hierarchy: pd.DataFrame,
           pred_hierarchy: pd.DataFrame,
           age_patterns: pd.DataFrame,
           population: pd.Series,
           age_specific_population: pd.Series,
           durations: Dict,
           day_inflection: pd.Timestamp,
           day_0: pd.Timestamp,
           pred_start_date: pd.Timestamp,
           pred_end_date: pd.Timestamp,
           num_threads: int,
           progress_bar: bool,
           **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cumulative_deaths = epi_data['cumulative_deaths'].dropna()
    daily_deaths = epi_data['daily_deaths'].dropna()
    ifr_age_pattern = age_patterns['ifr']
    sero_age_pattern = age_patterns['seroprevalence']

    ratio_data_scalar = variants.condition_out_variants(
        sero_location_dates=(seroprevalence.loc[seroprevalence['is_outlier'] == 0,
                                                ['location_id', 'date']]
                             .values.tolist()),
        hierarchy=mr_hierarchy,
        daily_infections=daily_infections,
        variant_prevalence=variant_prevalence,
        variant_risk_ratios=variant_risk_ratios,
        exposure_to_seroconversion=durations['exposure_to_seroconversion'],
        seroconversion_to_measure=durations['seropositive_to_death'],
    )

    model_data = data.create_model_data(
        cumulative_deaths=cumulative_deaths,
        daily_deaths=daily_deaths,
        seroprevalence=seroprevalence,
        covariate_pool=covariate_pool,
        daily_infections=daily_infections,
        ratio_data_scalar=ratio_data_scalar,
        hierarchy=mr_hierarchy,
        population=population,
        day_0=day_0,
        durations=durations,
    )
    
    pred_data = data.create_pred_data(
        hierarchy=pred_hierarchy,
        covariate_pool=covariate_pool,
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
        age_spec_population=age_specific_population.copy(),
        mr_hierarchy=mr_hierarchy.copy(),
        pred_hierarchy=pred_hierarchy.copy(),
        day_0=day_0,
        day_inflection=day_inflection,
        covariate_list=list(covariate_pool),
        num_threads=num_threads,
        progress_bar=progress_bar,
    )
    
    lr_rr, hr_rr = age_standardization.get_risk_group_rr(
        ifr_age_pattern.copy(),
        sero_age_pattern.copy() ** 0,  # just use flat
        age_specific_population.copy(),
    )
    pred_lr = (pred * lr_rr).rename('pred_ifr_lr')
    pred_hr = (pred * hr_rr).rename('pred_ifr_hr')
    
    pred = pd.concat([pred, pred_lr, pred_hr], axis=1)

    pred = pred.rename(columns={
        'pred_ifr_lr': 'ifr_lr', 'pred_ifr_hr': 'ifr_hr', 'pred_ifr': 'ifr'
    })
    pred.loc[:, 'lag'] = durations["exposure_to_death"]
    
    return pred, model_data

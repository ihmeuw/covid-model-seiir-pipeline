from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.lib import math
from covid_model_seiir_pipeline.pipeline.fit.model.mrbrt import cascade
from covid_model_seiir_pipeline.pipeline.fit.model import age_standardization
from covid_model_seiir_pipeline.pipeline.fit.model.covariate_priors import (
    get_covariate_priors,
    get_covariate_constraints,
)


def run_model(model_data: pd.DataFrame,
              pred_data: pd.DataFrame,
              ihr_age_pattern: pd.Series,
              sero_age_pattern: pd.Series,
              age_spec_population: pd.Series,
              mr_hierarchy: pd.DataFrame,
              pred_hierarchy: pd.DataFrame,
              covariate_list: List[str],
              variant_risk_ratio: float) -> Tuple[Dict, Dict, pd.Series, pd.Series, pd.Series, pd.Series, Dict]:
    age_stand_scaling_factor = age_standardization.get_scaling_factor(
        ihr_age_pattern,
        sero_age_pattern,
        age_spec_population.loc[[1]],
        age_spec_population
    )
    model_data = model_data.set_index('location_id')
    model_data['ihr'] *= age_stand_scaling_factor[model_data.index]
    model_data = model_data.reset_index()
    
    model_data['logit_ihr'] = math.logit(model_data['ihr'])
    model_data['logit_ihr'] = model_data['logit_ihr'].replace((-np.inf, np.inf), np.nan)
    model_data['ihr_se'] = 1
    model_data['logit_ihr_se'] = 1
    model_data['intercept'] = 1
    
    # lose 0s and 1s
    model_data = model_data.loc[model_data['logit_ihr'].notnull()]
    
    covariate_priors = get_covariate_priors(1, 'ihr',)
    covariate_priors = {covariate: covariate_priors[covariate] for covariate in covariate_list}
    covariate_constraints = get_covariate_constraints('ihr', variant_risk_ratio,)
    covariate_constraints = {covariate: covariate_constraints[covariate] for covariate in covariate_list}
    covariate_lambdas_tight = {covariate: 1. for covariate in covariate_list}
    covariate_lambdas_loose = {covariate: 10. for covariate in covariate_list}

    var_args = {
        'dep_var': 'logit_ihr',
        'dep_var_se': 'logit_ihr_se',
        'fe_vars': ['intercept', 'variant_prevalence'] + covariate_list,
        'prior_dict': {
            **covariate_constraints,
        },
        're_vars': [],
        'group_var': 'location_id',
    }
    global_prior_dict = covariate_priors
    location_prior_dict = {}
    pred_replace_dict = {}
    pred_exclude_vars = []
    level_lambdas = {
        0: {'intercept':   2., 'variant_prevalence': 1., **covariate_lambdas_tight},  # G->SR
        1: {'intercept':   2., 'variant_prevalence': 1., **covariate_lambdas_tight},  # SR->R
        2: {'intercept': 100., 'variant_prevalence': 1., **covariate_lambdas_loose},  # R->A0
        3: {'intercept': 100., 'variant_prevalence': 1., **covariate_lambdas_loose},  # A0->A1
        4: {'intercept': 100., 'variant_prevalence': 1., **covariate_lambdas_loose},  # A1->A2
        5: {'intercept': 100., 'variant_prevalence': 1., **covariate_lambdas_loose},  # A2->A3
    }
    
    if var_args['group_var'] != 'location_id':
        raise ValueError('NRMSE data assignment assumes `study_id` == `location_id` (`location_id` must be group_var).')
    
    # SUPPRESSING CASCADE CONSOLE OUTPUT
    model_data_cols = ['location_id', 'date', var_args['dep_var'],
                       var_args['dep_var_se']] + var_args['fe_vars']
    model_data = model_data.loc[:, model_data_cols]
    model_data = model_data.dropna()
    mr_model_dict, prior_dicts = cascade.run_cascade(
        model_data=model_data.copy(),
        hierarchy=mr_hierarchy.copy(),
        var_args=var_args.copy(),
        global_prior_dict=global_prior_dict.copy(),
        location_prior_dict=location_prior_dict.copy(),
        level_lambdas=level_lambdas.copy(),
    )
    pred_data = pred_data.dropna()
    pred, pred_fe, pred_location_map = cascade.predict_cascade(
        pred_data=pred_data.copy(),
        hierarchy=pred_hierarchy.copy(),
        mr_model_dict=mr_model_dict.copy(),
        pred_replace_dict=pred_replace_dict.copy(),
        pred_exclude_vars=pred_exclude_vars.copy(),
        var_args=var_args.copy(),
        verbose=False,
    )
    
    pred = math.expit(pred).rename(pred.name.replace('logit_', ''))
    pred_fe = math.expit(pred_fe).rename(pred_fe.name.replace('logit_', ''))
    
    pred /= age_stand_scaling_factor
    pred_fe /= age_stand_scaling_factor
    
    pred = pred.clip(0, 1)
    pred_fe = pred_fe.clip(0, 1)

    return (mr_model_dict, prior_dicts, pred.dropna(), pred_fe.dropna(), pred_location_map,
            age_stand_scaling_factor, level_lambdas)

from typing import Dict, Iterable, Optional, Union, TYPE_CHECKING

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.pipeline.regression.model import (
    slime,
    reslime,
)

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.regression.specification import CovariateSpecification


def prep_regression_inputs(beta_fit: pd.Series,
                           covariates: pd.DataFrame,
                           hierarchy: pd.DataFrame):
    regression_inputs = pd.merge(beta_fit.dropna(), covariates, on=beta_fit.index.names)
    group_cols = ['super_region_id', 'region_id', 'location_id']
    regression_inputs = (regression_inputs
                         .merge(hierarchy[group_cols], on='location_id')
                         .reset_index()
                         .set_index(group_cols)
                         .sort_index())
    regression_inputs['intercept'] = 1.0
    regression_inputs['ln_beta'] = np.log(regression_inputs['beta'])
    return regression_inputs


def build_predictors_old(covariate_specs: Iterable['CovariateSpecification'],):
    covariate_models = []
    for covariate in covariates:
        cov_model = CovariateModel.from_specification(covariate)
        if prior_coefficients is not None and not cov_model.use_re:
            coefficient_val = prior_coefficients[covariate.name].mean()
            cov_model.gprior = np.array([coefficient_val, 1e-10])
        covariate_models[covariate.order].append(cov_model)
    ordered_covmodel_sets = [CovModelSet(covariate_group)
                             for _, covariate_group in sorted(covariate_models.items())]


def get_predictor(covariate: 'CovariateSpecification',
                  gaussian_priors: Dict[str, pd.DataFrame],
                  model: str,):
    if model == 'slime':
        model = slime.CovModel.from_specification(covariate)
    elif model == 'reslime':
        model = reslime.PredictorModel.from_specification(covariate)
        if model.original_prior == 'data':
            model.original_prior = gaussian_priors[covariate.name]
    else:
        raise
    return model


def build_predictors(regression_inputs: pd.DataFrame,
                     covariate_specs: Iterable['CovariateSpecification'],
                     gaussian_priors: Dict[str, pd.DataFrame],
                     prior_coefficients: Optional[pd.DataFrame],
                     model: str):
    predictors = []
    fixed_coefficients = []
    for covariate in covariate_specs:
        if np.all(regression_inputs[covariate.name] == 0):
            fixed_coefficients.append(
                pd.Series(
                    0.0,
                    name=covariate.name,
                    index=pd.Index(regression_inputs.reset_index()['location_id'].unique(), name='location_id'),
                )
            )
        elif prior_coefficients is not None and not covariate.group_level:
            coefficient_val = (
                prior_coefficients
                .reset_index()
                .set_index('location_id')[covariate.name]
                .drop_duplicates()
            )
            assert len(coefficient_val.index) == len(coefficient_val.index.drop_duplicates())
            fixed_coefficients.append(coefficient_val)
            coefficient_val = coefficient_val.reindex(regression_inputs.index, level='location_id')
            regression_inputs['ln_beta'] -= coefficient_val * regression_inputs[covariate.name]
        else:
            predictors.append(get_predictor(covariate, gaussian_priors, model))
    if model == 'slime':
        predictor_set = slime.CovModelSet(predictors)
    else:
        predictor_set = reslime.PredictorModelSet(predictors)
    return predictor_set, fixed_coefficients


def run_model(regression_inputs: pd.DataFrame,
              predictor_set: Union[slime.CovModelSet, reslime.PredictorModelSet],
              model: str):
    if model == 'slime':
        mr_data = slime.MRData(
            df=regression_inputs.reset_index(),
            col_group='location_id',
            col_obs='ln_beta',
            col_covs=[p.name for p in predictor_set]
        )
        regressor = slime.BetaRegressor(predictor_set)
        coefficients = regressor.fit(mr_data)
    else:
        mr_data = reslime.MRData(
            data=regression_inputs.reset_index(),
            response_column='ln_beta',
            predictors=[p.name for p in predictor_set],
            group_columns=['super_region_id', 'region_id', 'location_id'],
        )
        mr_model = reslime.MRModel(mr_data, predictor_set)
        coefficients = mr_model.fit_model()
    return coefficients


def run_beta_regression(beta_fit: pd.Series,
                        covariates: pd.DataFrame,
                        covariate_specs: Iterable['CovariateSpecification'],
                        gaussian_priors: Dict[str, pd.DataFrame],
                        prior_coefficients: Optional[pd.DataFrame],
                        hierarchy: pd.DataFrame,
                        model: str = 'reslime') -> pd.DataFrame:
    regression_inputs = prep_regression_inputs(
        beta_fit,
        covariates,
        hierarchy
    )
    predictor_set, fixed_coefficients = build_predictors(
        regression_inputs,
        covariate_specs,
        gaussian_priors,
        prior_coefficients,
        model,
    )
    coefficients = run_model(
        regression_inputs,
        predictor_set,
        model,
    )
    coefficients = pd.concat([coefficients] + fixed_coefficients, axis=1).reset_index()
    coefficients = (coefficients
                    .drop(columns=['super_region_id', 'region_id'])
                    .set_index('location_id'))

    return coefficients

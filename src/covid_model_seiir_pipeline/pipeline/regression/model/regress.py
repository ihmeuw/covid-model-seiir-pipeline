from typing import Dict, Iterable, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.pipeline.regression.model.reslime import (
    PredictorModel,
    PredictorModelSet,
    MRData,
    MRModel,
)

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.regression.specification import CovariateSpecification


def run_beta_regression(beta_fit: pd.Series,
                        covariates: pd.DataFrame,
                        covariate_specs: Iterable['CovariateSpecification'],
                        gaussian_priors: Dict[str, pd.DataFrame],
                        prior_coefficients: Optional[pd.DataFrame]) -> pd.DataFrame:
    regression_inputs = pd.merge(beta_fit.dropna(), covariates, on=beta_fit.index.names).sort_index()
    regression_inputs['ln_beta'] = np.log(regression_inputs['beta'])

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
            prior = gaussian_priors[covariate.name] if covariate.gprior == 'data' else covariate.gprior
            predictor = PredictorModel(
                predictor_name=covariate.name,
                group_level=covariate.group_level,
                bounds=covariate.bounds,
                gaussian_prior_params=prior,
            )
            predictors.append(predictor)

    mr_data = MRData(
        data=regression_inputs,
        response_column='ln_beta',
        predictors=[p.name for p in predictors],
        group_columns=list(set([p.group_level for p in predictors])),
    )
    predictor_set = PredictorModelSet(predictors)
    mr_model = MRModel(mr_data, predictor_set)
    coefficients = mr_model.fit_model()

    coefficients = pd.concat([coefficients] + fixed_coefficients, axis=1)

    return coefficients

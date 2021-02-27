from collections import defaultdict
import copy
from typing import Iterable, List, Optional, Union, TYPE_CHECKING

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.pipeline.regression.model.slime import (
    CovModel,
    CovModelSet,
    MRData,
    MRModel,
)

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.regression.specification import CovariateSpecification


class CovariateModel(CovModel):
    """Adapter around slime CovModel to translate covariate specs."""

    @classmethod
    def from_specification(cls, covariate: 'CovariateSpecification'):
        return cls(
            col_cov=covariate.name,
            use_re=covariate.use_re,
            bounds=np.array(covariate.bounds),
            gprior=np.array(covariate.gprior),
            re_var=covariate.re_var,
        )


class IBetaRegressor:

    def fit(self, mr_data: MRData, sequential_refit: bool) -> pd.DataFrame:
        raise NotImplementedError


class BetaRegressor(IBetaRegressor):

    def __init__(self, covmodel_set: CovModelSet):
        self.covmodel_set = covmodel_set
        self.col_covs = [covmodel.col_cov for covmodel in covmodel_set.cov_models]

    def fit_no_random(self, mr_data: MRData) -> np.ndarray:
        covmodel_set_fixed = copy.deepcopy(self.covmodel_set)
        for covmodel in covmodel_set_fixed.cov_models:
            covmodel.use_re = False
        mr_model_fixed = MRModel(mr_data, covmodel_set_fixed)
        mr_model_fixed.fit_model()
        return list(mr_model_fixed.result.values())[0]

    def fit(self, mr_data: MRData, _: bool = None) -> pd.DataFrame:
        mr_model = MRModel(mr_data, self.covmodel_set)
        mr_model.fit_model()
        cov_coef = mr_model.result
        coef = pd.DataFrame.from_dict(cov_coef, orient='index').reset_index()
        coef.columns = ['location_id'] + self.col_covs
        return coef


class BetaRegressorSequential(IBetaRegressor):

    def __init__(self, ordered_covmodel_sets, default_std=1.0):
        self.default_std = default_std
        self.ordered_covmodel_sets = copy.deepcopy(ordered_covmodel_sets)
        self.col_covs = []
        for covmodel_set in self.ordered_covmodel_sets:
            self.col_covs.extend([covmodel.col_cov for covmodel in covmodel_set.cov_models])

    def fit(self, mr_data, sequential_refit: bool) -> pd.DataFrame:
        covmodels = []
        covmodel_bounds = []
        covmodel_gprior_std = []
        while len(self.ordered_covmodel_sets) > 0:
            new_cov_models = self.ordered_covmodel_sets.pop(0).cov_models
            covmodel_set = CovModelSet(covmodels + new_cov_models)
            for cov_model in new_cov_models:
                covmodel_bounds.append(cov_model.bounds)
                covmodel_gprior_std.append(cov_model.gprior[1])
                cov_model.gprior[1] = np.inf

            regressor = BetaRegressor(covmodel_set)
            cov_coef_fixed = regressor.fit_no_random(mr_data)

            for covmodel, coef in zip(covmodel_set.cov_models[len(covmodels):],
                                      cov_coef_fixed[len(covmodels):]):
                covmodel.gprior[0] = coef
                covmodel.bounds = np.array([coef, coef])
            covmodels = covmodel_set.cov_models

        if sequential_refit:
            # Return the covariates to their original bounds and prior variance.
            for i, cov_model in enumerate(covmodels):
                cov_model.bounds = np.array(covmodel_bounds[i])
                cov_model.gprior[1] = covmodel_gprior_std[i]
            # Otherwise we'll do nothing and just refit the random effects.
        regressor = BetaRegressor(CovModelSet(covmodels))
        return regressor.fit(mr_data)


def prep_regression_inputs(beta_fit: pd.Series,
                           covariates: pd.DataFrame) -> MRData:
    """Convert inputs for the beta regression model."""
    join_cols = ['location_id', 'date']
    df = beta_df.merge(covariate_df, on=join_cols)
    df = df.loc[df['beta'] != 0]
    df = df.sort_values(by=join_cols)
    df['ln_beta'] = np.log(df['beta'])
    mrdata = MRData(df, col_group='location_id', col_obs='ln_beta', col_covs=cov_names)
    return mrdata


def build_regressor(covariates: Iterable['CovariateSpecification'],
                    prior_coefficients: Optional[pd.DataFrame]) -> Union[BetaRegressor, BetaRegressorSequential]:
    """
    Based on a list of `CovariateSpecification`s and an ordered list of lists of covariate
    names, create a CovModelSet.
    """
    # construct each CovModel independently. add to dict of list by covariate order
    covariate_models = defaultdict(list)
    for covariate in covariates:
        cov_model = CovariateModel.from_specification(covariate)
        if prior_coefficients is not None and not cov_model.use_re:
            coefficient_val = prior_coefficients[covariate.name].mean()
            cov_model.gprior = np.array([coefficient_val, 1e-10])
        covariate_models[covariate.order].append(cov_model)
    ordered_covmodel_sets = [CovModelSet(covariate_group)
                             for _, covariate_group in sorted(covariate_models.items())]
    if len(ordered_covmodel_sets) > 1:
        regressor = BetaRegressorSequential(ordered_covmodel_sets)
    else:
        regressor = BetaRegressor(ordered_covmodel_sets[0])

    return regressor

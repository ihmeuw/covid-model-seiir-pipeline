from argparse import ArgumentParser, Namespace
from collections import defaultdict
import logging
from pathlib import Path
import shlex
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from slime.model import CovModelSet, CovModel
from slime.core.data import MRData

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.specification import (RegressionSpecification,
                                                                 CovariateSpecification)
from covid_model_seiir_pipeline.regression.model import BetaRegressor, BetaRegressorSequential, predict, convolve_mean


log = logging.getLogger(__name__)


def align_beta_with_covariates(covariate_df: pd.DataFrame,
                               beta_df: pd.DataFrame,
                               cov_names: List[str]) -> MRData:
    """Convert inputs for the beta regression model."""
    join_cols = ['location_id', 'date']
    df = beta_df.merge(covariate_df, on=join_cols)
    df = df.loc[df['beta'] != 0]
    df = df.sort_values(by=join_cols)
    df['ln_beta'] = np.log(df['beta'])
    mrdata = MRData(df, col_group='location_id', col_obs='beta', col_covs=cov_names)
    return mrdata


def build_regressor(covariates: Iterable[CovariateSpecification]) -> Union[BetaRegressor, BetaRegressorSequential]:
    """
    Based on a list of `CovariateSpecification`s and an ordered list of lists of covariate
    names, create a CovModelSet.
    """
    # construct each CovModel independently. add to dict of list by covariate order
    covariate_models = defaultdict(list)
    for covariate in covariates:
        cov_model = CovModel(
            col_cov=covariate.name,
            use_re=covariate.use_re,
            bounds=np.array(covariate.bounds),
            gprior=np.array(covariate.gprior),
            re_var=covariate.re_var,
        )
        covariate_models[covariate.order].append(cov_model)
    ordered_covmodel_sets = [CovModelSet(covariate_group)
                             for _, covariate_group in sorted(covariate_models.items())]
    if len(ordered_covmodel_sets) > 1:
        regressor = BetaRegressorSequential(ordered_covmodel_sets)
    else:
        regressor = BetaRegressor(ordered_covmodel_sets[0])

    return regressor


def run_beta_regression(draw_id: int, regression_version: str) -> None:
    regression_specification: RegressionSpecification = RegressionSpecification.from_path(
        Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    )
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    location_ids = data_interface.load_location_ids()

    # -------------------------- LOAD INPUTS -------------------- #
    # The data we want to fit
    beta_df = data_interface.load_ode_fits(draw_id, location_ids)
    covariates = data_interface.load_covariates(regression_specification.covariates, location_ids)

    # -------------- BETA REGRESSION WITH LOADED COVARIATES -------------------- #
    mr_data = align_beta_with_covariates(covariates, beta_df, list(regression_specification.covariates))
    regressor = build_regressor(regression_specification.covariates.values())

    coefficients = regressor.fit(mr_data)
    data_interface.save_regression_coefficients(coefficients, draw_id)

    # Fixme: don't know why this is here.  Regression is done after coefficients.
    #  Review with forecast integration.
    prediction = predict(regressor, covariates, col_t='date', col_group='location_id')
    beta_pred = np.exp(prediction['ln_beta_pred']).values[None, :]
    beta_pred = convolve_mean(beta_pred, radius=[0, 0])
    prediction['beta_pred'] = beta_pred.ravel()

    keep_columns = ['location_id', 'date'] + list(regression_specification.covariates) + ['beta_pred']
    regression_betas = prediction[keep_columns]
    beta_fit_covariates = beta_df.merge(regression_betas, on=['location_id', 'date'], how='left')
    data_interface.save_regression_betas(beta_fit_covariates, draw_id)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--regression-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():

    args = parse_arguments()
    run_beta_regression(draw_id=args.draw_id, regression_version=args.regression_version)


if __name__ == '__main__':
    main()

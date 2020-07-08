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

from covid_model_seiir_pipeline.static_vars import REGRESSION_SPECIFICATION_FILE, INFECTION_COL_DICT
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.specification import (
    RegressionSpecification,
    CovariateSpecification
)
from covid_model_seiir_pipeline.regression.model import (
    ODEProcessInput,
    ODEProcess,
    BetaRegressor,
    BetaRegressorSequential,
    predict
)


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
    mrdata = MRData(df, col_group='location_id', col_obs='ln_beta', col_covs=cov_names)
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
    # Build helper abstractions
    regression_spec_file = Path(regression_version) / REGRESSION_SPECIFICATION_FILE
    regression_specification = RegressionSpecification.from_path(regression_spec_file)
    data_interface = RegressionDataInterface.from_specification(regression_specification)

    # Load data
    location_ids = data_interface.load_location_ids()
    location_data = data_interface.load_all_location_data(location_ids=location_ids,
                                                          draw_id=draw_id)
    covariates = data_interface.load_covariates(regression_specification.covariates, location_ids)

    # Run ODE fit
    np.random.seed(draw_id)
    beta_fit_inputs = ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        col_lag_days=INFECTION_COL_DICT['COL_ID_LAG'],
        col_observed=INFECTION_COL_DICT['COL_OBS_DEATHS'],
        alpha=regression_specification.parameters.alpha,
        sigma=regression_specification.parameters.sigma,
        gamma1=regression_specification.parameters.gamma1,
        gamma2=regression_specification.parameters.gamma2,
        solver_dt=regression_specification.parameters.solver_dt,
        day_shift=regression_specification.parameters.day_shift,
    )
    ode_model = ODEProcess(beta_fit_inputs)
    beta_fit = ode_model.process()

    # Run regression
    mr_data = align_beta_with_covariates(covariates, beta_fit, list(regression_specification.covariates))
    regressor = build_regressor(regression_specification.covariates.values())
    coefficients = regressor.fit(mr_data)
    prediction = predict(regressor, covariates, col_t='date', col_group='location_id', col_beta='ln_beta_pred')
    prediction['beta_pred'] = np.exp(prediction['ln_beta_pred'])

    # Format and save data.
    data_interface.save_regression_coefficients(coefficients, draw_id)

    keep_columns = ['location_id', 'date'] + list(regression_specification.covariates) + ['beta_pred']
    regression_betas = prediction[keep_columns]
    beta_fit_covariates = beta_fit.merge(regression_betas, on=['location_id', 'date'], how='left')
    data_interface.save_regression_betas(beta_fit_covariates, draw_id)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    draw_beta_params = ode_model.create_params_df()
    data_interface.save_draw_beta_param_file(draw_beta_params, draw_id)

    beta_start_end_dates = ode_model.create_start_end_date_df()
    data_interface.save_draw_date_file(beta_start_end_dates, draw_id)


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

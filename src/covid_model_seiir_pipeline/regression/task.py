from argparse import ArgumentParser, Namespace

import logging
from pathlib import Path
import shlex
from typing import Optional

import numpy as np

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.model import align_beta_with_covariates, build_regressor, predict


log = logging.getLogger(__name__)


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

    prediction = predict(regressor, covariates, col_t='date', col_group='location_id', col_beta='ln_beta_pred')
    prediction['beta_pred'] = np.exp(prediction['ln_beta_pred'])

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

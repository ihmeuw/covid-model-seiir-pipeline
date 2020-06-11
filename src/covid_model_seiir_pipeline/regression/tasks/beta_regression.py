from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from slime.model import CovModelSet
from slime.core.data import MRData
from covid_model_seiir.model_runner import ModelRunner

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.regression.covariate_model import convert_to_covmodel
from covid_model_seiir_pipeline.regression.specification import (RegressionSpecification,
                                                                 CovariateSpecification)
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


log = logging.getLogger(__name__)


def create_covariate_pool(draw_id: int,
                          location_ids: List[int],
                          covariates: List[CovariateSpecification],
                          regression_data_interface: RegressionDataInterface
                          ) -> pd.DataFrame:

    def _merge_helper(dfs, df) -> pd.DataFrame:
        date_col = static_vars.COVARIATE_COL_DICT['COL_DATE']
        if dfs.empty:
            dfs = df
        else:
            if date_col in df.columns and date_col in dfs.columns:
                dfs = dfs.merge(df, on=[static_vars.COVARIATE_COL_DICT['COL_LOC_ID'],
                                        static_vars.COVARIATE_COL_DICT['COL_DATE']])
            else:
                dfs = dfs.merge(df, on=[static_vars.COVARIATE_COL_DICT['COL_LOC_ID']])
        return dfs

    covariate_df = pd.DataFrame()
    covariate_set_scenarios: Dict[str, pd.DataFrame] = {}

    # iterate through covariates and pull in regression data and scenario data
    for covariate in covariates:
        if covariate.name == "intercept":
            continue

        # import covariates
        tmp_set = regression_data_interface.load_covariate(
            covariate=covariate.name,
            location_ids=location_ids,  # TODO: testing value
            draw_id=draw_id,
            use_draws=covariate.draws
        )

        df = tmp_set.pop(covariate.name)
        covariate_set_scenarios.update(tmp_set)

        # time dependent covariates versus not
        covariate_df = _merge_helper(covariate_df, df)

    # save covariates to disk for posterity
    regression_data_interface.save_covariates(covariate_df, draw_id)

    # create scenario set file
    scenario_df = pd.DataFrame()
    for df in covariate_set_scenarios.values():
        scenario_df = _merge_helper(scenario_df, df)
    regression_data_interface.save_scenarios(scenario_df, draw_id)

    return covariate_df


def convert_inputs_for_beta_model(covariate_df: pd.DataFrame, beta_df: pd.DataFrame,
                                  covmodel_set: CovModelSet) -> MRData:
    """
    Convert inputs for the beta regression model.

    :param data_cov: covariate specifications
    :param beta_df: data frame with beta outputs from the spline
    :param covmodel_set: set for a covariate model
    :return: MRData object
    """
    df = beta_df.merge(
        covariate_df,
        left_on=[static_vars.COL_DATE, static_vars.COL_GROUP],
        right_on=[static_vars.COVARIATE_COL_DICT['COL_DATE'],
                  static_vars.COVARIATE_COL_DICT['COL_LOC_ID']],
    )
    df = df.loc[df[static_vars.COL_BETA] != 0]
    df = df.sort_values(by=[static_vars.COL_GROUP, static_vars.COL_DATE])
    df['ln_' + static_vars.COL_BETA] = np.log(df[static_vars.COL_BETA])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]

    # quality check. shouldn't hit because we drop nulls in data_interface
    covs_na = []
    for name in cov_names:
        if name != static_vars.COL_INTERCEPT:
            if df[name].isna().values.any():
                covs_na.append(name)
    if len(covs_na) > 0:
        raise ValueError('NaN in covariate data: ' + str(covs_na))

    mrdata = MRData(df, col_group=static_vars.COL_GROUP, col_obs='ln_' + static_vars.COL_BETA,
                    col_covs=cov_names)

    return mrdata


def run_beta_regression(draw_id: int, regression_version: str) -> None:
    regress_spec: RegressionSpecification = RegressionSpecification.from_path(
        Path(regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    )
    ode_fit_spec: FitSpecification = FitSpecification.from_path(
        Path(regress_spec.data.ode_fit_version) / static_vars.FIT_SPECIFICATION_FILE
    )
    data_interface = RegressionDataInterface(
        regression_root=Path(regress_spec.data.output_root),
        covariate_root=Path(regress_spec.data.covariate_version),
        ode_fit_root=Path(ode_fit_spec.data.output_root),
        infection_root=Path(ode_fit_spec.data.infection_version),
        location_file=Path(ode_fit_spec.data.location_set_file)
    )
    location_ids = data_interface.load_location_ids()

    # -------------------------- LOAD INPUTS -------------------- #
    covariate_df = create_covariate_pool(
        draw_id=draw_id,
        location_ids=location_ids,
        covariates=list(regress_spec.covariates.values()),
        regression_data_interface=data_interface
    )
    beta_df = data_interface.load_ode_fits(location_ids, draw_id)

    # -------------- BETA REGRESSION WITH LOADED COVARIATES -------------------- #
    # Convert inputs for beta regression
    mr = ModelRunner()

    # create covariate model
    ordered_covmodel_sets, all_covmodels_set = convert_to_covmodel(
        list(regress_spec.covariates.values())
    )

    mr_data = convert_inputs_for_beta_model(covariate_df, beta_df, all_covmodels_set)
    # TODO: add coefficient version
    fixed_coefficients = None
    # fit beta regression
    mr.fit_beta_regression_prod(
        ordered_covmodel_sets=ordered_covmodel_sets,
        mr_data=mr_data,
        path=str(data_interface.regression_paths.get_draw_coefficient_file(draw_id)),
        df_cov_coef=fixed_coefficients,
        add_intercept=False,
    )

    # Forecast the beta forward with those coefficients
    regression_fit = data_interface.load_mr_coefficients(draw_id=draw_id)
    forecasts = mr.predict_beta_forward_prod(
        covmodel_set=all_covmodels_set,
        df_cov=covariate_df,
        df_cov_coef=regression_fit,
        col_t=static_vars.COVARIATE_COL_DICT['COL_DATE'],
        col_group=static_vars.COVARIATE_COL_DICT['COL_LOC_ID']
    )
    regression_betas = forecasts[
        [static_vars.COVARIATE_COL_DICT['COL_LOC_ID'],
         static_vars.COVARIATE_COL_DICT['COL_DATE']] +
        [c.col_cov for c in all_covmodels_set.cov_models] + ['beta_pred']
    ]
    beta_fit_covariates = beta_df.merge(
        regression_betas,
        left_on=[static_vars.INFECTION_COL_DICT['COL_LOC_ID'],
                 static_vars.INFECTION_COL_DICT['COL_DATE']],
        right_on=[static_vars.COVARIATE_COL_DICT['COL_LOC_ID'],
                  static_vars.COVARIATE_COL_DICT['COL_DATE']],
        how='left'
    )
    data_interface.save_regression_betas(beta_fit_covariates, draw_id, location_ids)


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

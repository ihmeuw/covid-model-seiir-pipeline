from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.regression.specification import (RegressionSpecification,
                                                           CovariateSpecification)
from seiir_model_pipeline.regression.data import RegressionDataInterface
from seiir_model_pipeline.globals import COVARIATE_COL_DICT

log = logging.getLogger(__name__)


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


def create_covariate_pool(draw_id: int,
                          covariates: List[CovariateSpecification],
                          regression_data_interface: RegressionDataInterface
                          ) -> pd.DataFrame:

    def _merge_helper(dfs, df) -> pd.DataFrame:
        if dfs.empty:
            dfs = df
        else:
            if COVARIATE_COL_DICT['COL_DATE'] in dfs.columns:
                dfs = dfs.merge(df, on=[COVARIATE_COL_DICT['COL_LOC_ID'],
                                        COVARIATE_COL_DICT['COL_DATE']])
            else:
                dfs = dfs.merge(df, on=[COVARIATE_COL_DICT['COL_LOC_ID']])
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
            location_ids=[123],  # TODO: testing value
            draw_id=draw_id,
            use_draws=covariate.draws
        )

        regression_key = regression_data_interface.covariate_scenario_val.format(
            covariate=covariate, scenario="regression"
        )
        df = tmp_set.pop(regression_key)
        covariate_set_scenarios.update(tmp_set)

        # time dependent covariates versus not
        covariate_df = _merge_helper(covariate_df, df)

    # save covariates to disk for posterity
    regression_data_interface.save_covariates(covariate_df, draw_id)

    # create scenario set file
    scenario_df = pd.DataFrame()
    for df in covariate_set_scenarios.values():
        scenario_df = _merge_helper(scenario_df, df)

    # save covariates to disk for posterity
    regression_data_interface.save_scenarios(covariate_df, draw_id)

    return covariate_df


def run_beta_regression(draw_id: int, regression_version: str):
    regress_spec: RegressionSpecification = RegressionSpecification.from_path(
        Path(regression_version) / "regression_specification.yaml"
    )
    regression_data_interface = RegressionDataInterface(regress_spec.data)

    create_covariate_pool(draw_id, list(regress_spec.covariates.values()),
                          regression_data_interface)


def main():

    args = parse_arguments()
    run_beta_regression(draw_id=args.draw_id, regression_version=args.regression_version)


if __name__ == '__main__':
    main()

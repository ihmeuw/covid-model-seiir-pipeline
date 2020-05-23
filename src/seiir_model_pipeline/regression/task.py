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
                          ) -> Dict[str, pd.DataFrame]:
    covariate_set_scenarios: Dict[str, pd.DataFrame] = {}
    covariate_set_used: Dict[str, pd.DataFrame] = {}

    for covariate in covariates:
        if covariate.name == "intercept":
            continue
        input_file_name = covariate.get_input_file(covariate.scenario)
        output_file_name_future = covariate.get_output_file(covariate.scenario)
        output_file_name_regress = covariate.get_output_file("regress")
        tmp_set = regression_data_interface.load_raw_covariate_scenario(
            input_file_name=input_file_name,
            output_file_name_regress=output_file_name_regress,
            output_file_name_scenario=output_file_name_future,
            location_ids=[123],  # TODO: testing value
            draw_id=draw_id)

        covariate_set_used.update(tmp_set)
        covariate_set_scenarios.update(tmp_set)

        if covariate.alternate_scenarios is None:
            scenarios: List = []
        else:
            scenarios: List = covariate.alternate_scenarios

        for scenario in scenarios:
            input_file_name = covariate.get_input_file(scenario)
            output_file_name_future = covariate.get_output_file(scenario)
            output_file_name_regress = covariate.get_output_file("regress")

            tmp_set = regression_data_interface.load_raw_covariate_scenario(
                input_file_name=input_file_name,
                output_file_name_regress=output_file_name_regress,
                output_file_name_scenario=output_file_name_future,
                location_ids=[123],  # TODO: testing value
                draw_id=draw_id)

            # confirm same history
            regress_df_used = covariate_set_used[output_file_name_regress]
            regress_df_scenario = tmp_set[output_file_name_regress]
            if not regress_df_used.equals(regress_df_scenario):
                raise RuntimeError(
                    "Observed data is not exactly equal between covariates in covariate "
                    f"pool. Covariate: {covariate.name}. Used scenario: {covariate.scenario}"
                    f"Alternate scenarios: {covariate.alternate_scenarios}.")

            # add to scenario set
            covariate_set_scenarios.update(tmp_set)

    # save full scenario set to disk for use in future scenarios
    regression_data_interface.save_covariate_set(covariate_set_scenarios, draw_id)

    return covariate_set_used


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

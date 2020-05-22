from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set

import pandas as pd


from seiir_model_pipeline.ode_fit import FitSpecification
from seiir_model_pipeline.ode_fit.data import ODEDataInterface
from seiir_model_pipeline.regression.specification import (RegressionSpecification,
                                                           RegressionData)
from seiir_model_pipeline.regression.data import (RegressionDataInterface,
                                                  CovariateDataInterface)
from seiir_model_pipeline.globals import COVARIATE_COL_DICT


def create_covariate_pool(regression_specification: RegressionSpecification):

    # get ode spec so we know how many draws we ran with
    ode_specification: FitSpecification = FitSpecification.from_path(
        Path(regression_specification.data.ode_fit_version) / "fit_specification.yaml"
    )

    # constuct interface to raw covariate inputs and ode data
    cov_data = CovariateDataInterface(regression_specification.data)
    regress_data = RegressionDataInterface(regression_specification.data)
    ode_data = ODEDataInterface(ode_specification.data)

    location_ids = ode_data.load_location_ids()

    # get covariate groups we are etling
    covariate_groups: Set[str] = set()
    for covariate in regression_specification.covariates.keys():
        cov_group = cov_data.covariate_paths.get_covariate_group_from_covariate(covariate)
        if cov_group in covariate_groups:
            raise ValueError("multiple scenarios of the same covariate cannot be use in the "
                             "regression. Specify one and only one. Duplicate was "
                             f" {covariate}. Covariates were: "
                             f"{regression_specification.covariates.keys()}")
        else:
            covariate_groups.add(cov_group)

    if any([spec.draws for spec in regression_specification.covariates.values()]):

        # TODO: this loop over draw will likely become problematic
        for draw_id in range(ode_specification.parameters.n_draws):
            for covariate_group in covariate_groups:
                cov_set = cov_data.load_raw_covariate_set(covariate_group, location_ids,
                                                          draw_id)
                regress_data.save_covariate_set(cov_set, draw_id)

    else:
        for covariate_group in covariate_groups:
            cov_set = cov_data.load_raw_covariate_set(covariate_group, location_ids)
            regress_data.save_covariate_set(cov_set)

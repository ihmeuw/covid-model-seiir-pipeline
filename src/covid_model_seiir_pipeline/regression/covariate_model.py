from typing import List, Tuple, Dict

import numpy as np
from slime.model import CovModel, CovModelSet

from covid_model_seiir_pipeline.regression.specification import CovariateSpecification


def convert_to_covmodel(covariates: List[CovariateSpecification]
                        ) -> Tuple[List[CovModelSet], CovModelSet]:
    """
    Based on a list of `CovariateSpecification`s and an ordered list of lists of covariate
    names, create a CovModelSet.
    """

    # construct each CovModel independently. add to dict of list by covariate order
    cov_models = []
    cov_model_order_dict: Dict[int, List[CovModel]] = {}
    for covariate in covariates:

        cov_model = CovModel(
            col_cov=covariate.name,
            use_re=covariate.use_re,
            bounds=np.array(covariate.bounds),
            gprior=np.array(covariate.gprior),
            re_var=covariate.re_var,
        )
        cov_models.append(cov_model)
        ordered_cov_set = cov_model_order_dict.get(covariate.order, [])
        ordered_cov_set.append(cov_model)

        # do I need this line?
        cov_model_order_dict[covariate.order] = ordered_cov_set

    # constuct a CovModelSet for each order
    ordered_covmodel_sets = []
    cov_orders = list(cov_model_order_dict.keys())
    cov_orders.sort()
    for order in cov_orders:
        cov_model_set = CovModelSet(cov_model_order_dict[order])
        ordered_covmodel_sets.append(cov_model_set)

    # constuct a CovModelSet for all
    all_covmodels_set = CovModelSet(cov_models)
    return ordered_covmodel_sets, all_covmodels_set

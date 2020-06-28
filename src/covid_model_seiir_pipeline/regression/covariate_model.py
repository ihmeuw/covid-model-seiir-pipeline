from collections import defaultdict
import itertools
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
    all_covmodel_set = CovModelSet(itertools.chain(*covariate_models.values()))
    return ordered_covmodel_sets, all_covmodel_set

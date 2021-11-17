from covid_model_seiir_pipeline.pipeline.regression.specification import (
    REGRESSION_JOBS,
    RegressionSpecification,
    HospitalParameters,
)
from covid_model_seiir_pipeline.pipeline.regression.data import (
    RegressionDataInterface,
)
from covid_model_seiir_pipeline.pipeline.regression.task import (
    beta_regression,
    hospital_correction_factors,
)
from covid_model_seiir_pipeline.pipeline.regression.main import (
    do_beta_regression,
    regress,
)

SPECIFICATION = RegressionSpecification
COMMAND = regress
APPLICATION_MAIN = do_beta_regression
TASKS = {
    REGRESSION_JOBS.regression: beta_regression,
    REGRESSION_JOBS.hospital_correction_factors: hospital_correction_factors,
}

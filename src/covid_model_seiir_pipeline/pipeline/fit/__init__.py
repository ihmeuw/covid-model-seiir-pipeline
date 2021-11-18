from covid_model_seiir_pipeline.pipeline.fit.specification import (
    FIT_JOBS,
    FitSpecification,
)
from covid_model_seiir_pipeline.pipeline.fit.data import (
    FitDataInterface,
)
from covid_model_seiir_pipeline.pipeline.fit.task import (
    beta_fit,
)
from covid_model_seiir_pipeline.pipeline.fit.main import (
    do_fit,
    fit,
)

SPECIFICATION = FitSpecification
COMMAND = fit
APPLICATION_MAIN = do_fit
TASKS = {
    FIT_JOBS.beta_fit: beta_fit,
}

from covid_model_seiir_pipeline.pipeline.fit.specification import (
    FIT_JOBS,
    FitSpecification,
)
from covid_model_seiir_pipeline.pipeline.fit.data import (
    FitDataInterface,
)
from covid_model_seiir_pipeline.pipeline.fit.task import (
    covariate_pool,
    beta_fit,
    beta_fit_postprocess,
    beta_fit_diagnostics,
)
from covid_model_seiir_pipeline.pipeline.fit.main import (
    do_fit,
    fit,
)

SPECIFICATION = FitSpecification
COMMAND = fit
APPLICATION_MAIN = do_fit
TASKS = {
    FIT_JOBS.covariate_pool: covariate_pool,
    FIT_JOBS.beta_fit: beta_fit,
    FIT_JOBS.beta_fit_postprocess: beta_fit_postprocess,
    FIT_JOBS.beta_fit_diagnostics: beta_fit_diagnostics,
}

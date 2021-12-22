from covid_model_seiir_pipeline.side_analysis.parameter_fit.data import (
    FitDataInterface
)
from covid_model_seiir_pipeline.side_analysis.parameter_fit.specification import (
    FIT_JOBS,
    FitSpecification,
)
from covid_model_seiir_pipeline.side_analysis.parameter_fit.task import (
    ode_parameter_fit,
)
from covid_model_seiir_pipeline.side_analysis.parameter_fit.main import (
    do_parameter_fit,
    parameter_fit,
)

SPECIFICATION = FitSpecification
COMMAND = parameter_fit
APPLICATION_MAIN = do_parameter_fit
TASKS = {
    FIT_JOBS.fit: ode_parameter_fit
}

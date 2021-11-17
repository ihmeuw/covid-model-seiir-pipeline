from covid_model_seiir_pipeline.pipeline.diagnostics.specification import (
    DIAGNOSTICS_JOBS,
    DiagnosticsSpecification,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.task import (
    cumulative_deaths_compare_csv,
    grid_plots,
    scatters,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.main import (
    do_diagnostics,
    diagnostics,
)

SPECIFICATION = DiagnosticsSpecification
COMMAND = diagnostics
APPLICATION_MAIN = do_diagnostics
TASKS = {
    DIAGNOSTICS_JOBS.scatters: scatters,
    DIAGNOSTICS_JOBS.grid_plots: grid_plots,
    DIAGNOSTICS_JOBS.compare_csv: cumulative_deaths_compare_csv,
}

from pathlib import Path

from covid_shared import cli_tools, ihme_deps
from loguru import logger

from covid_model_seiir_pipeline.lib import static_vars
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import DiagnosticsSpecification
from covid_model_seiir_pipeline.pipeline.diagnostics.workflow import DiagnosticsWorkflow


def do_diagnostics(app_metadata: cli_tools.Metadata,
                   diagnostics_specification: DiagnosticsSpecification,
                   preprocess_only: bool):
    logger.info(f'Starting diagnostics for version {diagnostics_specification.data.output_root}.')

    output_root = Path(diagnostics_specification.data.output_root)
    diagnostics_specification.dump(output_root / static_vars.DIAGNOSTICS_SPECIFICATION_FILE)

    if not preprocess_only:
        workflow = DiagnosticsWorkflow(diagnostics_specification.data.output_root,
                                       diagnostics_specification.workflow)
        grid_plot_jobs = [grid_plot_spec.name for grid_plot_spec in diagnostics_specification.grid_plots]

        workflow.attach_tasks(grid_plot_jobs)

        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')

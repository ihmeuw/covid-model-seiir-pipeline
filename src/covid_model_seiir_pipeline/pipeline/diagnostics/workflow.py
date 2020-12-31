import shutil
from typing import List

from covid_model_seiir_pipeline.lib import workflow
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import (
    DIAGNOSTICS_JOBS,
)


class GridPlotsTaskTemplate(workflow.TaskTemplate):

    task_name_template = f"{DIAGNOSTICS_JOBS.grid_plots}_{{name}}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{DIAGNOSTICS_JOBS.grid_plots} " +
            "--name {name} "
    )
    node_args = ['name']


class DiagnosticsWorkflow(workflow.WorkflowTemplate):
    workflow_name_template = 'seiir-diagnostics-{version}'
    task_template_classes = {
        DIAGNOSTICS_JOBS.grid_plots: GridPlotsTaskTemplate,
    }
    # Jobs here are not homogeneous so it's useful to get all failures if
    # things do fail.
    fail_fast = False

    def attach_tasks(self, grid_plots_task_names: List[str]) -> None:
        grid_plots_template = self.task_templates[DIAGNOSTICS_JOBS.grid_plots]

        for grid_plots_task_name in grid_plots_task_names:
            grid_plots_task = grid_plots_template.get_task(
                name=grid_plots_task_name,
            )
            self.workflow.add_task(grid_plots_task)

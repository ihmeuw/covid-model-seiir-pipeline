import shutil

from covid_model_seiir_pipeline.lib import workflow
from covid_model_seiir_pipeline.pipeline.regression.specification import REGRESSION_JOBS


class BetaRegressionTaskTemplate(workflow.TaskTemplate):
    task_name_template = f"{REGRESSION_JOBS.regression}_draw_{{draw_id}}"
    command_template = (
            f"{shutil.which('stask')} " +
            f"{REGRESSION_JOBS.regression} "
            "--regression-version {regression_version} " +
            "--draw-id {draw_id} "
            "-vv"
    )
    node_args = ['draw_id']
    task_args = ['regression_version']


class RegressionWorkflow(workflow.WorkflowTemplate):

    workflow_name_template = 'seiir-regression-{version}'
    task_template_classes = {
        REGRESSION_JOBS.regression: BetaRegressionTaskTemplate
    }

    def attach_tasks(self, n_draws: int):
        regression_template = self.task_templates[REGRESSION_JOBS.regression]
        for draw_id in range(n_draws):
            task = regression_template.get_task(
                regression_version=self.version,
                draw_id=draw_id
            )
            self.workflow.add_task(task)

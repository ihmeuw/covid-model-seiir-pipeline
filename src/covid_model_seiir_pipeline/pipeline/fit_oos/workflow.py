import shutil
from typing import Dict

from covid_model_seiir_pipeline.lib import workflow
from covid_model_seiir_pipeline.pipeline.fit_oos.specification import FIT_JOBS, FitScenario


class BetaFitTaskTemplate(workflow.TaskTemplate):
    task_name_template = f"{FIT_JOBS.fit}_{{scenario}}_{{draw_id}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{FIT_JOBS.fit} "
        "--fit-version {forecast_version} "
        "--scenario {scenario} "
        "--draw-id {draw_id} "
        "-vv"
    )
    node_args = ['scenario', 'draw_id']
    task_args = ['fit_version']


class FitWorkflow(workflow.WorkflowTemplate):

    workflow_name_template = 'seiir-oos-fit-{version}'
    task_template_classes = {
        FIT_JOBS.fit: BetaFitTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, scenarios: Dict[str, FitScenario]):
        fit_template = self.task_templates[FIT_JOBS.forecast]

        for scenario_name, scenario_spec in scenarios.items():
            for draw in range(n_draws):
                fit_task = fit_template.get_task(
                    fit_version=self.version,
                    draw_id=draw,
                    scenario=scenario_name,
                )
                self.workflow.add_task(fit_task)

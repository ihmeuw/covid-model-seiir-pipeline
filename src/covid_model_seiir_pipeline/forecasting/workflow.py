from itertools import product
from typing import List

from jobmon.client.swarm.executors.base import ExecutorParameters

from covid_model_seiir_pipeline.workflow_template import TaskTemplate, WorkflowTemplate



class BetaForecastTaskTemplate(TaskTemplate):

    task_name_template = "forecast_{scenario}_{draw_id}"
    command_template = (
        "beta_forecast " +
        "--draw-id {draw_id} " +
        "--forecast-version {forecast_version} " +
        "--scenario {scenario}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=1000,
        m_mem_free='3G',
        num_cores=1,
        queue='d.q'
    )


class ForecastWorkflow(WorkflowTemplate):

    workflow_name_template = 'seiir-forecast-{version}'
    task_templates = {'forecast': BetaForecastTaskTemplate}

    def attach_tasks(self, n_draws: int, scenarios: List[str]):
        forecast_template = self.task_templates['forecast']
        for draw, scenario in product(range(n_draws), scenarios):
            task = forecast_template.get_task(
                forecast_version=self.version,
                draw_id=draw,
                scenario=scenario
            )
            self.workflow.add_task(task)

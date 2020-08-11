from itertools import product
from typing import List

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities

FORECAST_RUNTIME = 1000
FORECAST_MEMORY = '5G'
FORECAST_CORES = 1
FORECAST_SCALING_CORES = 50
FORECAST_QUEUE = 'd.q'


class BetaForecastTaskTemplate:

    def __init__(self):
        self.command_template = (
            "beta_forecast " +
            "--draw-id {draw_id} " +
            "--forecast-version {forecast_version} " +
            "--scenario {scenario}"
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=FORECAST_RUNTIME,
            m_mem_free=FORECAST_MEMORY,
            num_cores=FORECAST_CORES,
            queue=FORECAST_QUEUE,
        )

    @classmethod
    def get_task_name(cls, draw_id: int, scenario: str) -> str:
        return f"forecast_{scenario}_{draw_id}"

    def get_task(self, draw_id: int, forecast_version: str, scenario: str):
        task_name = self.get_task_name(draw_id, scenario)
        task = BashTask(
            command=self.command_template.format(draw_id=draw_id,
                                                 forecast_version=forecast_version,
                                                 scenario=scenario),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1
        )
        return task


class ForecastWorkflow:

    def __init__(self, version: str) -> None:
        self.version = version
        self.task_template = BetaForecastTaskTemplate()

        workflow_args = f'seiir-forecast-{version}'
        stdout, stderr = utilities.make_log_dirs(version, prefix='jobmon')

        self.workflow = Workflow(
            workflow_args=workflow_args,
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_scenario_tasks(self, n_draws: int, scenarios: List[str]):
        for draw, scenario in product(range(n_draws), scenarios):
            task = self.task_template.get_task(draw, self.version, scenario)
            self.workflow.add_task(task)

    def run(self):
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")

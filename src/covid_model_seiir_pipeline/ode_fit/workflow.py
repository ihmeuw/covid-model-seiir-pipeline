from typing import Dict

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities


class ODEFitTaskTemplate:

    def __init__(self):
        self.task_registry: Dict[str, BashTask] = {}
        self.command_template = (
            "beta_ode_fit " +
            "--draw-id {draw_id} " +
            "--ode-version {ode_version} "
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=3000,
            m_mem_free='20G',
            num_cores=3,
            queue='d.q'
        )

    @staticmethod
    def get_task_name(draw_id: int) -> str:
        return f"ode_fit_{draw_id}"

    def get_task(self, draw_id: int, ode_version: str):
        task_name = self.get_task_name(draw_id)
        task = BashTask(
            command=self.command_template.format(draw_id=draw_id,
                                                 ode_version=ode_version),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1
        )
        self.task_registry[task_name] = task
        return task


class ODEFitWorkflow:

    def __init__(self, version: str):
        self.version = version
        self._ode_fit_task_template = ODEFitTaskTemplate()

        workflow_args = f'seiir-model-{version}'
        stdout, stderr = utilities.make_log_dirs(version, prefix='jobmon')

        self.workflow = Workflow(
            workflow_args=workflow_args,
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_ode_tasks(self, n_draws: int):
        for draw_id in range(n_draws):
            task = self._ode_fit_task_template.get_task(draw_id, self.version)
            self.workflow.add_task(task)

    def run(self):
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")

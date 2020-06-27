from typing import Dict

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities


class BetaRegressionTaskTemplate:

    def __init__(self, task_registry: Dict[str, BashTask]):
        self.task_registry = task_registry
        self.command_template = (
            "beta_regression " +
            "--draw-id {draw_id} " +
            "--regression-version {regression_version} "
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=1000,
            m_mem_free='20G',
            num_cores=3,
            queue='d.q'
        )

    @staticmethod
    def get_task_name(draw_id: int) -> str:
        return f"regression_{draw_id}"

    def get_task(self, draw_id: int, regression_version: str):
        task_name = self.get_task_name(draw_id)
        task = BashTask(
            command=self.command_template.format(draw_id=draw_id,
                                                 regression_version=regression_version),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1
        )
        self.task_registry[task_name] = task
        return task


class RegressionDiagnosticsTaskTemplate:

    def __init__(self, task_registry: Dict[str, BashTask]):
        self.task_registry = task_registry
        self.command_template = (
            "regression_diagnostics " +
            "--regression-version {regression_version} "
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=1000,
            m_mem_free='10G',
            num_cores=3,
            queue='d.q'
        )

    @staticmethod
    def get_task_name() -> str:
        return "regression_diagnostics"

    def get_task(self, regression_version: str):
        # get dependencies
        upstream_deps_names = [k for k in self.task_registry.keys() if "regression_" in k]
        upstream_deps = [self.task_registry[k] for k in upstream_deps_names]

        # construct task
        task_name = self.get_task_name()
        task = BashTask(
            command=self.command_template.format(regression_version=regression_version),
            name=task_name,
            executor_parameters=self.params,
            max_attempts=1,
            upstream_tasks=upstream_deps
        )
        self.task_registry[task_name] = task
        return task


class RegressionWorkflow:

    def __init__(self, regression_version):
        self.regression_version = regression_version
        # if we need to build dependencies then the task registry must be shared between
        # task factories
        self.task_registry: Dict[str, BashTask] = {}
        self._beta_regression_tt = BetaRegressionTaskTemplate(self.task_registry)
        self._regression_diagnostics_tt = RegressionDiagnosticsTaskTemplate(self.task_registry)

        workflow_args = f'seiir-regression-{regression_version}'
        stdout, stderr = utilities.make_log_dirs(
            regression_version, prefix='jobmon'
        )

        self.workflow = Workflow(
            workflow_args=workflow_args,
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_beta_regression_tasks(self, n_draws: int):
        for draw_id in range(n_draws):
            task = self._beta_regression_tt.get_task(
                draw_id, self.regression_version
            )
            self.workflow.add_task(task)

    def attach_regression_diagnostics_task(self):
        task = self._regression_diagnostics_tt.get_task(
            self.regression_version
        )
        self.workflow.add_task(task)

    def run(self):
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")

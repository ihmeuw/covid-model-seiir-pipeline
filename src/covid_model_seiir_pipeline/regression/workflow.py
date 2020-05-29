from typing import Dict

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters

from covid_model_seiir_pipeline import utilities
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification


class RegressionTaskTemplate:

    def __init__(self):
        self.task_registry: Dict[str, BashTask] = {}
        self.command_template = (
            "beta_regression " +
            "--draw-id {draw_id} " +
            "--regression-version {regression_version} "
        )
        self.params = ExecutorParameters(
            max_runtime_seconds=1000,
            j_resource=False,
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


class RegressionWorkflow:

    def __init__(self, regression_specification: RegressionSpecification,
                 ode_fit_specification: FitSpecification):
        self.regression_specification = regression_specification
        self.ode_fit_specification = ode_fit_specification

        # if we need to build dependencies then the task registry must be shared between
        # task factories
        self._regression_task_template = RegressionTaskTemplate()

        workflow_args = f'seiir-regression-{regression_specification.data.output_root}'
        stdout, stderr = utilities.make_log_dirs(
            self.regression_specification.data.output_root, prefix='jobmon'
        )

        self.workflow = Workflow(
            workflow_args=workflow_args,
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_regression_tasks(self):
        for draw_id in range(self.ode_fit_specification.parameters.n_draws):
            task = self._regression_task_template.get_task(
                draw_id, self.regression_specification.data.output_root
            )
            self.workflow.add_task(task)

    def run(self):
        self.workflow.run()

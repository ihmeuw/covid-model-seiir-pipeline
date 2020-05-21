from typing import Dict

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters

from seiir_model_pipeline import utilities
from seiir_model_pipeline.ode_fit import FitSpecification


class ODEFitTaskTemplate:

    def __init__(self):
        self.task_registry: Dict[str, BashTask] = {}
        self.command_template = (
            "beta_ode_fit " +
            "--draw-id {draw_id} " +
            "--ode-version {ode_version} "
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

    def __init__(self, fit_specification: FitSpecification):
        self.fit_specification = fit_specification

        # if we need to build dependencies then the task registry must be shared between
        # task factories
        self._ode_fit_task_template = ODEFitTaskTemplate()

        # TODO: figure out where to put these defaults
        # TODO: configure logging to go to output_root
        workflow_args = f'seiir-model-{fit_specification.data.output_root}'
        stdout, stderr = utilities.make_log_dirs(self.fit_specification.data.output_root,
                                                 prefix='jobmon')

        self.workflow = Workflow(
            workflow_args=workflow_args,
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_ode_tasks(self):
        for draw_id in range(self.fit_specification.parameters.n_draws):
            task = self._ode_fit_task_template.get_task(draw_id, self.fit_specification.data.output_root)
            self.workflow.add_task(task)

    def run(self):
        self.workflow.run()

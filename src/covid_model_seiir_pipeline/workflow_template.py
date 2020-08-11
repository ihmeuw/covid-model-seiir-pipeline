import abc
from typing import Dict

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities


class TaskTemplate:

    task_name_template: str = None
    command_template: str = None
    params: ExecutorParameters = None

    @classmethod
    def get_task(cls, *_, **kwargs) -> BashTask:
        task = BashTask(
            command=cls.command_template.format(**kwargs),
            name=cls.task_name_template.format(**kwargs),
            executor_parameters=cls.params,
            max_attempts=1
        )
        return task


class WorkflowTemplate:

    workflow_name_template: str = None
    task_templates: Dict[str, TaskTemplate] = None

    def __init__(self, version: str):
        self.version = version
        stdout, stderr = utilities.make_log_dirs(version, prefix='jobmon')

        self.workflow = Workflow(
            workflow_args=self.workflow_name_template.format(version=version),
            project="proj_covid",
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    @abc.abstractmethod
    def attach_tasks(self, *args, **kwargs):
        pass

    def run(self):
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")


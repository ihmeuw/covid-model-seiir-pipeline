import abc
from typing import Dict

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters as ExecutorParameters_
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities

DEFAULT_PROJECT = 'proj_dq'
DEFAULT_QUEUE = 'd.q'


class ExecutorParameters(ExecutorParameters_):

    def __init__(self,
                 queue: str,
                 max_runtime_seconds: int = None,
                 m_mem_free: str = None,
                 num_cores: int = None,
                 **kwargs):
        if max_runtime_seconds is None:
            max_runtime_seconds = self.default_runtime
        if m_mem_free is None:
            m_mem_free = self.default_memory
        if num_cores is None:
            num_cores = self.default_cores
        super().__init__(num_cores=num_cores,
                         queue=queue,
                         max_runtime_seconds=max_runtime_seconds,
                         m_mem_free=m_mem_free,
                         **kwargs)

    @abc.abstractmethod
    @property
    def default_runtime(self) -> int:
        ...

    @abc.abstractmethod
    @property
    def default_memory(self) -> str:
        ...

    @abc.abstractmethod
    @property
    def default_cores(self) -> int:
        ...

    def to_dict(self):
        return self.to_wire()


class WorkflowSpecification:

    def __init__(self, project: str, task_specifications: Dict[str, ExecutorParameters]):
        self.project = project
        self.task_specifications = task_specifications

    def to_dict(self):
        return {
            'project': self.project,
            'task_specifications': {
                task: spec.to_dict() for task, spec in self.task_specifications.items(),
            }
        }


class TaskTemplate:
    """Abstract factory for producing SEIIR pipeline tasks."""

    def __init__(self,
                 task_name_template: str,
                 command_template: str,
                 params: ExecutorParameters,
                 max_attempts: int = 1):
        self.task_name_template = task_name_template
        self.command_template = command_template
        self.params = params
        self.max_attempts = max_attempts

    def get_task(self, *_, **kwargs) -> BashTask:
        task = BashTask(
            command=self.command_template.format(**kwargs),
            name=self.task_name_template.format(**kwargs),
            executor_parameters=self.params,
            max_attempts=self.max_attempts
        )
        return task


class WorkflowTemplate:

    def __init__(self, version: str, workflow_spec: WorkflowSpecification):
        self.version = version

        stdout, stderr = utilities.make_log_dirs(version, prefix='jobmon')

        self.workflow = Workflow(
            workflow_args=self.workflow_name_template.format(version=version),
            project=workflow_spec.project,
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

        self._task_templates = self.build_task_templates(workflow_spec.task_specifications)


    @abc.abstractmethod
    @property
    def workflow_name_template(self) -> str:
        ...

    @abc.abstractmethod
    def build_task_templates(self, task_specifications: Dict[str, ExecutorParameters]):
        ...

    @abc.abstractmethod
    def attach_tasks(self, *args, **kwargs):
        pass

    @property
    def task_templates(self) -> Dict[str, TaskTemplate]:
        return self._task_templates

    def run(self):
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")


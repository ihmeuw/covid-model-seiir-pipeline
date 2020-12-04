import abc
from dataclasses import dataclass, field
import re
from typing import Dict

from loguru import logger
from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities


DEFAULT_PROJECT = 'proj_covid'
DEFAULT_QUEUE = 'd.q'


class TaskSpecification(abc.ABC):
    name: str
    default_max_runtime_seconds: int
    default_m_mem_free: str
    default_num_cores: int

    def __init__(self, task_specification_dict: Dict):
        runtime_bounds = [100, 60*60*24]
        self.max_runtime_seconds: int = task_specification_dict.pop('max_runtime_seconds',
                                                                    self.default_max_runtime_seconds)
        if not runtime_bounds[0] <= self.max_runtime_seconds <= runtime_bounds[1]:
            raise ValueError(f'Invalid max runtime for {self.name}: {self.max_runtime_seconds}. '
                             f'Max runtime must be in the range {runtime_bounds}.')

        mem_re = re.compile('\\d+[G]')
        mem_bounds_gb = [1, 1000]
        self.m_mem_free: str = task_specification_dict.pop('m_mem_free',
                                                      self.default_m_mem_free)
        if not mem_re.match(self.m_mem_free):
            raise ValueError('Memory request is expected to be in the format "XG" where X is an integer. '
                             f'You provided {self.m_mem_free} for {self.name}.')
        if not mem_bounds_gb[1] <= int(self.m_mem_free[:-1]) <= mem_bounds_gb[1]:
            raise ValueError(f'Invalid max memory for {self.name}: {self.m_mem_free}. ',
                             f'Max memory must be in the range {mem_bounds_gb}G.')

        num_core_bounds = [1, 30]
        self.num_cores: int = task_specification_dict.pop('num_cores',
                                                     self.default_num_cores)
        if not num_core_bounds[0] <= self.num_cores <= num_core_bounds[1]:
            raise ValueError(f'Invalid num cores for {self.name}: {self.num_cores}. '
                             f'Num cores must be in the range {num_core_bounds}')

        allowed_queues = ['d.q', 'all.q', 'long.q']
        self.queue: str = task_specification_dict.pop('queue', DEFAULT_QUEUE)
        if self.queue not in allowed_queues:
            raise ValueError(f'Invalid queue for {self.name}: {self.queue}. '
                             f'Queue must be one of {allowed_queues}.')

        if task_specification_dict:
            logger.warning(f'Unknown task options specified for {self.name}: {task_specification_dict}. '
                           f'These options will be ignored.')

    def to_dict(self):
        return {'max_runtime_seconds': self.max_runtime_seconds,
                'm_mem_free': self.m_mem_free,
                'num_cores': self.num_cores,
                'queue': self.queue}


class TaskTemplate(abc.ABC):

    task_name_template: str = None
    command_template: str = None

    def __init__(self, task_specification: TaskSpecification):
        self.params = ExecutorParameters(**task_specification.to_dict())

    def get_task(self, *_, **kwargs) -> BashTask:
        task = BashTask(
            command=self.command_template.format(**kwargs),
            name=self.task_name_template.format(**kwargs),
            executor_parameters=self.params,
            max_attempts=1
        )
        return task


@dataclass
class WorkflowSpecification:
    project: str = field(default='proj_covid')
    tasks: Dict[str, TaskSpecification] = field(default_factory=dict)

    def to_dict(self):
        return {'project': self.project,
                'tasks': {k: v.to_dict() for k, v in self.tasks}}


class WorkflowTemplate(abc.ABC):

    workflow_name_template: str = None
    task_templates: Dict[str, TaskTemplate] = None

    def __init__(self, version: str, workflow_specification: WorkflowSpecification):
        self.version = version
        self.task_templates = self.build_task_templates(workflow_specification.tasks)
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
    def build_task_templates(self, task_specifications: Dict[str, TaskSpecification]) -> Dict[str, TaskTemplate]:
        pass

    @abc.abstractmethod
    def attach_tasks(self, *args, **kwargs) -> None:
        pass

    def run(self) -> None:
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")


"""Primitives for construction jobmon workflows and workflow specifications."""

import abc
import re
from typing import Dict, Type, TypeVar, Union

from loguru import logger
from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities


DEFAULT_PROJECT = 'proj_covid'
DEFAULT_QUEUE = 'd.q'

_TaskSpecDict = Dict[str, Union[str, int]]


class TaskSpecification(abc.ABC):
    """Validation wrapper for specifying task execution parameters.

    This class is intended to outline the parameter space for task
    specification and provide sanity checks on the passed parameters so
    that we fail fast and informatively.

    Subclasses are meant to inherit and provide default values for the
    class variables. At runtime those defaults will be used to fill values
    not provided in the specification file when the specification file
    is parsed and then the full task specification will be validated against
    the values in the `validate` method of this class.

    """
    default_max_runtime_seconds: int
    default_m_mem_free: str
    default_num_cores: int

    def __init__(self, task_specification_dict: _TaskSpecDict):
        self.name = self.__class__.__name__
        self.max_runtime_seconds: int = task_specification_dict.pop('max_runtime_seconds',
                                                                    self.default_max_runtime_seconds)
        self.m_mem_free: str = task_specification_dict.pop('m_mem_free',
                                                           self.default_m_mem_free)
        self.num_cores: int = task_specification_dict.pop('num_cores',
                                                          self.default_num_cores)
        # Workflow specification guarantees this will be present.
        self.queue = task_specification_dict.pop('queue')

        if task_specification_dict:
            logger.warning(f'Unknown task options specified for {self.name}: {task_specification_dict}. '
                           f'These options will be ignored.')

    def validate(self):
        """Checks specification against some baseline constraints."""
        runtime_bounds = [100, 60*60*24]
        if not runtime_bounds[0] <= self.max_runtime_seconds <= runtime_bounds[1]:
            raise ValueError(f'Invalid max runtime for {self.name}: {self.max_runtime_seconds}. '
                             f'Max runtime must be in the range {runtime_bounds}.')

        mem_re = re.compile('\\d+[G]')
        mem_bounds_gb = [1, 1000]
        if not mem_re.match(self.m_mem_free):
            raise ValueError('Memory request is expected to be in the format "XG" where X is an integer. '
                             f'You provided {self.m_mem_free} for {self.name}.')
        if not mem_bounds_gb[0] <= int(self.m_mem_free[:-1]) <= mem_bounds_gb[1]:
            raise ValueError(f'Invalid max memory for {self.name}: {self.m_mem_free}. ',
                             f'Max memory must be in the range {mem_bounds_gb}G.')

        num_core_bounds = [1, 30]
        if not num_core_bounds[0] <= self.num_cores <= num_core_bounds[1]:
            raise ValueError(f'Invalid num cores for {self.name}: {self.num_cores}. '
                             f'Num cores must be in the range {num_core_bounds}')

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """Coerce the specification to a dict for display or write to disk."""
        return {'max_runtime_seconds': self.max_runtime_seconds,
                'm_mem_free': self.m_mem_free,
                'num_cores': self.num_cores}

    def __repr__(self):
        return f'{self.name}({", ".join([f"{k}={v}" for k, v in self.to_dict().items()])})'


TTaskSpecification = TypeVar('TTaskSpecification', bound=TaskSpecification)


class TaskTemplate(abc.ABC):
    """Factory class for a parameterized task.

    Subclasses are intended to inherit and provide string templates for the
    class variables ``task_name_template`` which will construct cluster
    job names from the task args and ``command_template`` which will
    resolve the task args into a job executable by bash.

    """

    task_name_template: str
    command_template: str

    def __init__(self, task_specification: TaskSpecification):
        self.params = ExecutorParameters(**task_specification.to_dict())

    def get_task(self, *_, **kwargs) -> BashTask:
        """Resolve job arguments into a bash executable task for jobmon."""
        task = BashTask(
            command=self.command_template.format(**kwargs),
            name=self.task_name_template.format(**kwargs),
            executor_parameters=self.params,
            max_attempts=1
        )
        return task


TTaskTemplate = TypeVar('TTaskTemplate', bound=TaskTemplate)


class WorkflowSpecification(abc.ABC):
    """Validation wrapper for specifying workflow execution parameters.

    This class is intended to outline the parameter space for workflow
    specification and provide sanity checks on the passed parameters so
    that we fail fast and informatively.

    Subclasses are meant to inherit and provide a dict mapping task type
    names to concrete subclasses of the ``TaskSpecification`` base class
    as the ``tasks`` class variable.  When instantiated, the
    ``WorkflowSpecification`` subclasses will automatically resolve provided
    specification values with defaults and do some sanity check validation.

    """

    tasks: Dict[str, Type[TTaskSpecification]]

    def __init__(self,
                 tasks: Dict[str, Dict[str, Union[int, str]]],
                 project: str = None,
                 queue: str = None):
        self.name: str = self.__class__.__name__
        self.project: str = project if project is not None else DEFAULT_PROJECT
        self.queue: str = queue if queue is not None else DEFAULT_QUEUE

        # Check everything's okay before making the task specs
        self.validate()

        self.task_specifications: Dict[str, TaskSpecification] = self.process_task_dicts(tasks)

    def validate(self):
        """Checks specification against some baseline constraints."""
        allowed_projects = ['proj_dq', 'proj_covid', 'proj_covid_prod']
        if self.project not in allowed_projects:
            raise ValueError(f'Invalid project selected for {self.name}: {self.project}. '
                             f'Covid workflows should be run on one of {allowed_projects}.')
        allowed_queues = ['d.q', 'all.q', 'long.q']
        if self.queue not in allowed_queues:
            raise ValueError(f'Invalid queue for {self.name}: {self.queue}. '
                             f'Queue must be one of {allowed_queues}.')

    def process_task_dicts(self,
                           task_specification_dicts: Dict[str, _TaskSpecDict]) -> Dict[str, TaskSpecification]:
        """Converts parsed input specifications for tasks into concrete
        instances of ``TaskSpecification`` subclasses.

        """
        task_specifications = {}
        for task_name, spec_class in self.tasks.items():
            task_spec_dict = task_specification_dicts.pop(task_name, {})
            task_spec_dict['queue'] = self.queue
            task_specifications[task_name] = spec_class(task_spec_dict)

        if task_specification_dicts:
            logger.warning(f'Task specifications provided for unknown tasks: {list(task_specification_dicts)}.'
                           f'These specifications will be ignored.')

        return task_specifications

    def to_dict(self) -> Dict:
        """Coerce the specification to a dict for display or write to disk."""
        return {'project': self.project,
                'queue': self.queue,
                'tasks': {k: v.to_dict() for k, v in self.task_specifications.items()}}

    def __repr__(self):
        return f'{self.name}({", ".join([f"{k}={v}" for k, v in self.to_dict().items()])})'


class WorkflowTemplate(abc.ABC):
    """Factory for building and running workflows from specifications.

    Subclasses are intended to inherit and provide a string template for the
    class variable ``workflow_name_template`` which takes an output version
    string and maps it to a unique workflow name, and a dictionary mapping
    task type names to concrete subclasses of the ``TaskTemplate`` base
    class as the ``task_template_classes`` class variable. This mapping
    is tightly coupled with the ``tasks`` mapping of the associated
    workflow specification (ie, they should have exactly the same keys).

    When instantiated, the ``WorkflowTemplate`` subclasses will
    produce a jobmon workflow from the provided workflow specification,
    set up output logging directories, and produce concrete templates for
    workflow tasks.

    Subclass implementers must provide an implementation of ``attach_tasks``
    which takes relevant model parameters as arguments and builds and attaches
    an appropriate task dag to the jobmon workflow.

    """

    workflow_name_template: str = None
    task_template_classes: Dict[str, Type[TTaskTemplate]]

    def __init__(self, version: str, workflow_specification: WorkflowSpecification):
        self.version = version
        assert workflow_specification.tasks.keys() == self.task_template_classes.keys()
        self.task_templates = self.build_task_templates(workflow_specification.tasks)
        stdout, stderr = utilities.make_log_dirs(version, prefix='jobmon')

        self.workflow = Workflow(
            workflow_args=self.workflow_name_template.format(version=version),
            project=workflow_specification.project,
            stderr=stderr,
            stdout=stdout,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def build_task_templates(self, task_specifications: Dict[str, TaskSpecification]) -> Dict[str, TaskTemplate]:
        """Parses task specifications into task templates."""
        task_templates = {}
        for task_name, task_specification in task_specifications.items():
            task_templates[task_name] = self.task_template_classes[task_name](task_specification.to_dict())
        return task_templates

    @abc.abstractmethod
    def attach_tasks(self, *args, **kwargs) -> None:
        """Turn model arguments into jobmon workflow tasks."""
        pass

    def run(self) -> None:
        """Execute the constructed workflow."""
        execution_status = self.workflow.run()
        if execution_status != DagExecutionStatus.SUCCEEDED:
            raise RuntimeError("Workflow failed. Check database or logs for errors")


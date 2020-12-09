"""Primitives for construction jobmon workflow specifications."""
import abc
import re
from typing import Dict, Type, TypeVar, Union

from loguru import logger


DEFAULT_PROJECT = 'proj_covid'
DEFAULT_QUEUE = 'd.q'

_TaskSpecDict = Dict[str, Union[str, int]]


class TaskSpecification:
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
    # Class variables meant to be overridden by subclasses.
    default_max_runtime_seconds: int
    default_m_mem_free: str
    default_num_cores: int

    # Constants for all tasks.
    _runtime_bounds = [100, 60 * 60 * 24]
    _mem_re = re.compile('\\d+[G]')
    _mem_bounds_gb = [1, 1000]
    _num_core_bounds = [1, 30]

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

    def __init_subclass__(cls, **kwargs):
        name = cls.__name__

        for attr in ['default_max_runtime_seconds', 'default_m_mem_free', 'default_num_cores']:
            if getattr(cls, attr) is None:
                raise AttributeError(f'No default value provided for {attr} for task template subclass {name}. '
                                     'Check your class definition.')

        if not cls._runtime_bounds[0] <= cls.default_max_runtime_seconds <= cls._runtime_bounds[1]:
            raise AttributeError(f'Invalid default max runtime for {name}: {cls.default_max_runtime_seconds}. '
                                 f'Max runtime must be in the range {cls._runtime_bounds}.'
                                 'Check your class definition.')

        if not cls._mem_re.match(cls.default_m_mem_free):
            raise AttributeError('Memory request is expected to be in the format "XG" where X is an integer. '
                                 f'You provided {cls.default_m_mem_free} for {name}.')
        if not cls._mem_bounds_gb[0] <= int(cls.default_m_mem_free[:-1]) <= cls._mem_bounds_gb[1]:
            raise AttributeError(f'Invalid default max memory for {name}: {cls.default_m_mem_free}. ',
                                 f'Max memory must be in the range {cls._mem_bounds_gb}G. '
                                 'Check your class definition.')

        if not cls._num_core_bounds[0] <= cls.default_num_cores <= cls._num_core_bounds[1]:
            raise AttributeError(f'Invalid num cores for {name}: {cls.default_num_cores}. '
                                 f'Num cores must be in the range {cls._num_core_bounds}l '
                                 'Check your class definition.')

    def validate(self):
        """Checks specification against some baseline constraints."""
        if not self._runtime_bounds[0] <= self.max_runtime_seconds <= self._runtime_bounds[1]:
            raise ValueError(f'Invalid max runtime for {self.name}: {self.max_runtime_seconds}. '
                             f'Max runtime must be in the range {self._runtime_bounds}.')

        if not self._mem_re.match(self.m_mem_free):
            raise ValueError('Memory request is expected to be in the format "XG" where X is an integer. '
                             f'You provided {self.m_mem_free} for {self.name}.')
        if not self._mem_bounds_gb[0] <= int(self.m_mem_free[:-1]) <= self._mem_bounds_gb[1]:
            raise ValueError(f'Invalid max memory for {self.name}: {self.m_mem_free}. ',
                             f'Max memory must be in the range {self._mem_bounds_gb}G.')

        if not self._num_core_bounds[0] <= self.num_cores <= self._num_core_bounds[1]:
            raise ValueError(f'Invalid num cores for {self.name}: {self.num_cores}. '
                             f'Num cores must be in the range {self._num_core_bounds}')

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """Coerce the specification to a dict for display or write to disk."""
        return {'max_runtime_seconds': self.max_runtime_seconds,
                'm_mem_free': self.m_mem_free,
                'num_cores': self.num_cores,
                'queue': self.queue}

    def __repr__(self):
        return f'{self.name}({", ".join([f"{k}={v}" for k, v in self.to_dict().items()])})'


TTaskSpecification = TypeVar('TTaskSpecification', bound=TaskSpecification)


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
                 tasks: Dict[str, Dict[str, Union[int, str]]] = None,
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

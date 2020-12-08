"""Primitives for construction jobmon workflows."""
import abc
from typing import Dict, Type, TypeVar

from jobmon.client import Workflow, BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters
from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus

from covid_model_seiir_pipeline import utilities
from covid_model_seiir_pipeline.workflow_tools.specification import (
    TaskSpecification,
    WorkflowSpecification
)


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
        self.task_templates = self.build_task_templates(workflow_specification.task_specifications)
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
            task_templates[task_name] = self.task_template_classes[task_name](task_specification)
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

"""Wrappers for IHME specific dependencies.

This module explicitly declares and wraps IHME specific code bases
to prevent CI failures at import time.
"""
import importlib


def _lazy_import_callable(module_path: str, object_name: str):
    """Delays import errors for callables until called."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, object_name)
    except ModuleNotFoundError:
        def f(*args, **kwargs):
            raise ModuleNotFoundError(f"No module named '{module_path}'. Cannot find {object_name}.")
        return f


try:
    from db_queries import get_location_metadata
except ModuleNotFoundError:
    get_location_metadata = _lazy_import_callable('db_queries', 'get_location_metadata')

try:
    from jobmon.client.api import (
        Tool,
        ExecutorParameters,
    )
    from jobmon.client.task import Task
    from jobmon.client.workflow import WorkflowRunStatus
    from jobmon.exceptions import WorkflowAlreadyComplete
except ModuleNotFoundError:
    Tool = _lazy_import_callable('jobmon.client.api', 'Tool')
    ExecutorParameters = _lazy_import_callable('jobmon.client.api', 'ExecutorParameters')
    Task = _lazy_import_callable('jobmon.client.task', 'BashTask')
    WorkflowRunStatus = _lazy_import_callable('jobmon.client.workflow', 'WorkflowRunStatus')
    WorkflowAlreadyComplete = _lazy_import_callable('jobmon.exceptions', 'WorkflowAlreadyComplete')

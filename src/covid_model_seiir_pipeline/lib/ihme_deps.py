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
            raise ModuleNotFoundError(f"No module named '{module_path}'")
        return f


get_location_metadata = _lazy_import_callable('db_queries', 'get_location_metadata')

Workflow = _lazy_import_callable('jobmon.client', 'Workflow')
BashTask = _lazy_import_callable('jobmon.client', 'BashTask')
ExecutorParameters = _lazy_import_callable('jobmon.client.swarm.executors.base', 'ExecutorParameters')
DagExecutionStatus = _lazy_import_callable('jobmon.client.swarm.workflow.task_dag', 'DagExecutionStatus')
WorkflowAlreadyComplete = _lazy_import_callable('jobmon.client.swarm.workflow.workflow', 'WorkflowAlreadyComplete')

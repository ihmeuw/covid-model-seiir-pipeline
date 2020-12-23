
def _lazy_import_error_object(module_name: str):

    class C:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(f"No module named '{module_name}'")


def _lazy_import_error_function(module_name: str):
    def f(*args, **kwargs):
        raise ModuleNotFoundError(f"No module named '{module_name}'")


try:
    from db_queries import get_location_metadata
except ModuleNotFoundError:
    get_location_metadata = _lazy_import_error_function('db_queries')

try:
    from jobmon.client import Workflow, BashTask
    from jobmon.client.swarm.executors.base import ExecutorParameters
    from jobmon.client.swarm.workflow.task_dag import DagExecutionStatus
    from jobmon.client.swarm.workflow.workflow import WorkflowAlreadyComplete
except ModuleNotFoundError:
    Workflow = _lazy_import_error_function('jobmon')
    BashTask = _lazy_import_error_function('jobmon')
    ExecutorParameters = _lazy_import_error_function('jobmon')
    DagExecutionStatus = _lazy_import_error_function('jobmon')
    WorkflowAlreadyComplete = _lazy_import_error_function('jobmon')

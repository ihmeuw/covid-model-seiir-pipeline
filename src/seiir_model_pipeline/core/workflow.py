import getpass
import logging

from seiir_model_pipeline.core.task import RegressionTask, ForecastTask
from jobmon.client.swarm.workflow.workflow import Workflow

log = logging.getLogger(__name__)

PROJECT = 'proj_fake_project'


class SEIIRWorkFlow(Workflow):
    def __init__(self, directories):
        """
        Create a workflow for the SEIIR pipeline.

        :param directories: (Directories)
        :return (Workflow)
        """
        self.directories = directories

        user = getpass.getuser()
        working_dir = f'/ihme/homes/{user}'
        workflow_args = f'seiir-model-'
        if directories.regression_version is not None:
            workflow_args += f'{directories.regression_version} '
        if directories.forecast_version is not None:
            workflow_args += f'{directories.forecast_version} '

        super().__init__(
            workflow_args=workflow_args,
            project=PROJECT,
            stderr=str(directories.log_dir),
            stdout=str(directories.log_dir),
            working_dir=working_dir,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_regression_tasks(self, n_draws, **kwargs):
        """
        Attach n_draws DrawTasks.

        :param n_draws: (int)
        **kwargs: keyword arguments to DrawTask
        :return: self
        """
        tasks = [RegressionTask(draw_id=i, **kwargs) for i in range(n_draws)]
        self.add_tasks(tasks)
        return tasks

    def attach_forecast_tasks(self, location_ids, n_draws, **kwargs):
        tasks = [
            ForecastTask(location_id=loc, draw_id=i, **kwargs)
            for loc in location_ids for i in range(n_draws)
        ]
        self.add_tasks(tasks)
        return tasks

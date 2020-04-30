import getpass
import logging

from seiir_model_pipeline.core.task import DrawTask
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
        workflow_args = f'seiir-model-{directories.output_version}'

        super().__init__(
            workflow_args=workflow_args,
            project=PROJECT,
            stderr=str(directories.error_dir),
            stdout=str(directories.output_dir),
            working_dir=working_dir,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_tasks(self, n_draws, **kwargs):
        """
        Attach n_draws DrawTasks.

        :param n_draws: (int)
        **kwargs: keyword arguments to DrawTask
        :return: self
        """
        for i in range(n_draws):
            self.add_tasks(
                DrawTask(draw_id=i, **kwargs)
            )
        return self

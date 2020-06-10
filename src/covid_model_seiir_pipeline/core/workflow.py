import getpass
import logging

from covid_model_seiir_pipeline.core.task import RegressionTask, ForecastTask, ScalingDiagnosticTask
from covid_model_seiir_pipeline.core.task import RegressionDiagnosticTask
from jobmon.client.swarm.workflow.workflow import Workflow

log = logging.getLogger(__name__)

PROJECT = 'proj_covid'


class SEIIRWorkFlow(Workflow):
    def __init__(self, directories):
        """
        Create a Jobmon workflow for the SEIIR pipeline.

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

        # TODO: right now my scratch directory is where the logs
        #  are saving -- I was getting errors when writing to sgeoutput
        super().__init__(
            workflow_args=workflow_args,
            project=PROJECT,
            stderr=f'/share/temp/sgeoutput/{user}/errors',
            stdout=f'/share/temp/sgeoutput/{user}/output',
            working_dir=working_dir,
            seconds_until_timeout=60*60*24,
            resume=True
        )

    def attach_regression_tasks(self, n_draws, add_diagnostic, **kwargs):
        """
        Attach n_draws RegressionTasks and adds a diagnostic task if needed. Will load
        coefficients from a previous run to fix them.

        :param n_draws: (int)
        :param add_diagnostic: (bool) add a diagnostic task
        **kwargs: keyword arguments to DrawTask
        :return: self
        """
        tasks = [RegressionTask(draw_id=i, **kwargs)
                 for i in range(n_draws)]
        self.add_tasks(tasks)
        if add_diagnostic:
            diagnostic_task = RegressionDiagnosticTask(**kwargs)
            self.add_task(diagnostic_task)
            for t in tasks:
                diagnostic_task.add_upstream(t)
        return tasks

    def attach_forecast_tasks(self, location_ids, add_splicer, add_diagnostic, **kwargs):
        """
        Attach a forecast task for each location ID (draws are done inside of each task),
        and optionally add a splicing task for each one and a diagnostic task for each one.

        :param location_ids: (List[int])
        :param add_splicer: (bool)
        :param add_diagnostic: (bool)
        :param kwargs: keyword arguments to ForecastTask
        :return:
        """
        splicer_tasks = []
        tasks = [
            ForecastTask(location_id=loc, **kwargs)
            for loc in location_ids
        ]
        if add_splicer:
            for task in tasks:
                splicer_tasks = splicer_tasks + task.add_splicer_task(add_diagnostic)
        self.add_tasks(tasks)
        self.add_tasks(splicer_tasks)

        if add_diagnostic:
            scaling_task = ScalingDiagnosticTask(**kwargs)
            for task in tasks:
                scaling_task.add_upstream(task)
            self.add_task(scaling_task)

        return tasks + splicer_tasks

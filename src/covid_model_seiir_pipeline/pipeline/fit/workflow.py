import shutil

from covid_shared import workflow

import covid_model_seiir_pipeline
from covid_model_seiir_pipeline.pipeline.fit.specification import FIT_JOBS


class CovariatePoolTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{FIT_JOBS.covariate_pool}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{FIT_JOBS.covariate_pool} "
            "--fit-version {fit_version} "
            "-vv"
    )
    node_args = []
    task_args = ['fit_version']


class BetaFitTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{FIT_JOBS.beta_fit}_draw_{{draw_id}}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{FIT_JOBS.beta_fit} "
            "--fit-version {fit_version} "
            "--draw-id {draw_id} "
            "-vv"
    )
    node_args = ['draw_id']
    task_args = ['fit_version']


class FitWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-fit-{version}'
    task_template_classes = {
        FIT_JOBS.covariate_pool: CovariatePoolTaskTemplate,
        FIT_JOBS.beta_fit: BetaFitTaskTemplate,
    }

    def attach_tasks(self, n_draws: int):
        covariate_template = self.task_templates[FIT_JOBS.covariate_pool]
        fit_template = self.task_templates[FIT_JOBS.beta_fit]

        covariate_pool_task = covariate_template.get_task(fit_version=self.version)
        self.workflow.add_task(covariate_pool_task)
        for draw_id in range(n_draws):
            task = fit_template.get_task(
                fit_version=self.version,
                draw_id=draw_id,
            )
            task.add_upstream(covariate_pool_task)
            self.workflow.add_task(task)

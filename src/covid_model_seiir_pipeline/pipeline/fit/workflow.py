import shutil
from typing import List

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


class JoinSentinelTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{FIT_JOBS.beta_fit_join_sentinel}_{{sentinel_id}}"
    command_template = (
        "echo 'join sentinel {fit_version} {sentinel_id}'; sleep 1"
    )
    node_args = ['sentinel_id']
    task_args = ['fit_version']


class BetaFitPostprocessTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{FIT_JOBS.beta_fit_postprocess}_measure_{{measure}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{FIT_JOBS.beta_fit_postprocess} "
        "--fit-version {fit_version} "
        "--measure {measure} "
        "-vv"
    )
    node_args = ['measure']
    task_args = ['fit_version']


class FitWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-fit-{version}'
    task_template_classes = {
        FIT_JOBS.covariate_pool: CovariatePoolTaskTemplate,
        FIT_JOBS.beta_fit: BetaFitTaskTemplate,
        FIT_JOBS.beta_fit_join_sentinel: JoinSentinelTaskTemplate,
        FIT_JOBS.beta_fit_postprocess: BetaFitPostprocessTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, measures: List[str]):
        covariate_template = self.task_templates[FIT_JOBS.covariate_pool]
        fit_template = self.task_templates[FIT_JOBS.beta_fit]
        join_template = self.task_templates[FIT_JOBS.beta_fit_join_sentinel]
        postprocess_template = self.task_templates[FIT_JOBS.beta_fit_postprocess]

        covariate_pool_task = covariate_template.get_task(fit_version=self.version)
        self.workflow.add_task(covariate_pool_task)
        join_task = join_template.get_task(fit_version=self.version, sentinel_id='postprocess')
        self.workflow.add_task(join_task)
        for draw_id in range(n_draws):
            task = fit_template.get_task(
                fit_version=self.version,
                draw_id=draw_id,
            )
            task.add_upstream(covariate_pool_task)
            task.add_downstream(join_task)
            self.workflow.add_task(task)

        for measure in measures:
            task = postprocess_template.get_task(
                fit_version=self.version,
                measure=measure,
            )
            task.add_upstream(join_task)

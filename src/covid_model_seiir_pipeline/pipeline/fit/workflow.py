import itertools
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
    task_name_template = f"{FIT_JOBS.beta_fit}_measure_{{measure}}_draw_{{draw_id}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{FIT_JOBS.beta_fit} "
        "--fit-version {fit_version} "
        "--measure {measure} "
        "--draw-id {draw_id} "
        "-vv"
    )
    node_args = ['measure', 'draw_id']
    task_args = ['fit_version']


class PastInfectionsTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{FIT_JOBS.past_infections}_draw_{{draw_id}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{FIT_JOBS.past_infections} "
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
        "echo join sentinel {fit_version} {sentinel_id}"
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


class BetaFitDiagnosticsTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{FIT_JOBS.beta_fit_diagnostics}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{FIT_JOBS.beta_fit_diagnostics} "
        "--fit-version {fit_version} "   
        "--plot-type {plot_type} "
        "-vv"
    )
    node_args = ['plot_type']
    task_args = ['fit_version']


class FitWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-fit-{version}'
    task_template_classes = {
        FIT_JOBS.covariate_pool: CovariatePoolTaskTemplate,
        FIT_JOBS.beta_fit: BetaFitTaskTemplate,
        FIT_JOBS.past_infections: PastInfectionsTaskTemplate,
        FIT_JOBS.beta_fit_join_sentinel: JoinSentinelTaskTemplate,
        FIT_JOBS.beta_fit_postprocess: BetaFitPostprocessTaskTemplate,
        FIT_JOBS.beta_fit_diagnostics: BetaFitDiagnosticsTaskTemplate,
    }
    fail_fast = True

    def attach_tasks(self, n_draws: int, measures: List[str], plot_types: List[str]):
        covariate_template = self.task_templates[FIT_JOBS.covariate_pool]
        fit_template = self.task_templates[FIT_JOBS.beta_fit]
        past_infections_template = self.task_templates[FIT_JOBS.past_infections]
        join_template = self.task_templates[FIT_JOBS.beta_fit_join_sentinel]
        postprocess_template = self.task_templates[FIT_JOBS.beta_fit_postprocess]
        diagnostics_template = self.task_templates[FIT_JOBS.beta_fit_diagnostics]

        covariate_pool_task = covariate_template.get_task(
            fit_version=self.version,
        )
        self.workflow.add_task(covariate_pool_task)
        fit_join_task = join_template.get_task(
            fit_version=self.version,
            sentinel_id='fit',
        )
        self.workflow.add_task(fit_join_task)
        for measure, draw_id in itertools.product(['case', 'death', 'admission'], range(n_draws)):
            task = fit_template.get_task(
                fit_version=self.version,
                measure=measure,
                draw_id=draw_id,
            )
            task.add_upstream(covariate_pool_task)
            task.add_downstream(fit_join_task)
            self.workflow.add_task(task)

        past_infections_join_task = join_template.get_task(
            fit_version=self.version,
            sentinel_id='past_infections',
        )
        self.workflow.add_task(fit_join_task)
        for draw_id in range(n_draws):
            task = past_infections_template.get_task(
                fit_version=self.version,
                draw_id=draw_id
            )
            task.add_upstream(fit_join_task)
            task.add_downstream(past_infections_join_task)

        postprocess_join_task = join_template.get_task(
            fit_version=self.version,
            sentinel_id='postprocess',
        )
        self.workflow.add_task(postprocess_join_task)
        for measure in measures:
            task = postprocess_template.get_task(
                fit_version=self.version,
                measure=measure,
            )
            task.add_upstream(past_infections_join_task)
            task.add_downstream(postprocess_join_task)
            self.workflow.add_task(task)

        for plot_type in plot_types:
            diagnostics_task = diagnostics_template.get_task(
               fit_version=self.version,
               plot_type=plot_type,
            )
            diagnostics_task.add_upstream(postprocess_join_task)
            self.workflow.add_task(diagnostics_task)

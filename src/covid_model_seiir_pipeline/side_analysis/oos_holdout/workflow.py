import shutil
from typing import List

from covid_shared import workflow

import covid_model_seiir_pipeline
from covid_model_seiir_pipeline.side_analysis.oos_holdout.specification import OOS_HOLDOUT_JOBS


class OOSRegressionTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{OOS_HOLDOUT_JOBS.oos_regression}_draw_{{draw_id}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{OOS_HOLDOUT_JOBS.oos_regression} "
        "--oos-holdout-version {oos_holdout_version} "
        "--draw-id {draw_id} "
        "-vv"
    )
    node_args = ['draw_id']
    task_args = ['oos_holdout_version']


class OOSBetaScalingTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{OOS_HOLDOUT_JOBS.oos_beta_scaling}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{OOS_HOLDOUT_JOBS.oos_beta_scaling} "
        "--oos-holdout-version {oos_holdout_version} "        
        "-vv"
    )
    node_args = []
    task_args = ['oos_holdout_version']


class OOSForecastTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{OOS_HOLDOUT_JOBS.oos_forecast}_draw_{{draw_id}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{OOS_HOLDOUT_JOBS.oos_forecast} "
        "--oos-holdout-version {oos_holdout_version} "
        "--draw-id {draw_id} "
        "-vv"
    )
    node_args = ['draw_id']
    task_args = ['oos_holdout_version']


class OOSJoinSentinelTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{OOS_HOLDOUT_JOBS.oos_join_sentinel}_{{sentinel_id}}"
    command_template = (
        "echo join sentinel {oos_holdout_version} {sentinel_id}"
    )
    node_args = ['sentinel_id']
    task_args = ['oos_holdout_version']


class OOSPostprocessTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{OOS_HOLDOUT_JOBS.oos_postprocess}_measure_{{measure}}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{OOS_HOLDOUT_JOBS.oos_postprocess} "
        "--oos-holdout-version {oos_holdout_version} "
        "--measure {measure} "
        "-vv"
    )
    node_args = ['measure']
    task_args = ['oos_holdout_version']


class OOSDiagnosticsTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{OOS_HOLDOUT_JOBS.oos_diagnostics}"
    command_template = (
        f"{shutil.which('stask')} "
        f"{OOS_HOLDOUT_JOBS.oos_diagnostics} "
        "--oos-holdout-version {oos_holdout_version} "   
        "--plot-type {plot_type} "
        "-vv"
    )
    node_args = ['plot_type']
    task_args = ['oos_holdout_version']


class OOSHoldoutWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-oos-{version}'
    task_template_classes = {
        OOS_HOLDOUT_JOBS.oos_regression: OOSRegressionTaskTemplate,
        OOS_HOLDOUT_JOBS.oos_beta_scaling: OOSBetaScalingTaskTemplate,
        OOS_HOLDOUT_JOBS.oos_forecast: OOSForecastTaskTemplate,
        OOS_HOLDOUT_JOBS.oos_join_sentinel: OOSJoinSentinelTaskTemplate,
        OOS_HOLDOUT_JOBS.oos_postprocess: OOSPostprocessTaskTemplate,
        OOS_HOLDOUT_JOBS.oos_diagnostics: OOSDiagnosticsTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, measures: List[str], plot_types: List[str]):
        regression_template = self.task_templates[OOS_HOLDOUT_JOBS.oos_regression]
        scaling_template = self.task_templates[OOS_HOLDOUT_JOBS.oos_beta_scaling]
        forecast_template = self.task_templates[OOS_HOLDOUT_JOBS.oos_forecast]
        join_template = self.task_templates[OOS_HOLDOUT_JOBS.oos_join_sentinel]
        postprocess_template = self.task_templates[OOS_HOLDOUT_JOBS.oos_postprocess]
        diagnostics_template = self.task_templates[OOS_HOLDOUT_JOBS.oos_diagnostics]

        scaling_task = scaling_template.get_task(oos_holdout_version=self.version)
        forecast_join_task = join_template.get_task(oos_holdout_version=self.version, sentinel_id='forecast')
        # postprocess_join_task = join_template.get_task(oos_holdout_version=self.version, sentinel_id='postprocess')
        for task in [scaling_task, forecast_join_task]:  # postprocess_join_task]:
            self.workflow.add_task(task)

        for draw_id in range(n_draws):
            regression_task = regression_template.get_task(
                oos_holdout_version=self.version,
                draw_id=draw_id,
            )
            regression_task.add_downstream(scaling_task)
            self.workflow.add_task(regression_task)

            forecast_task = forecast_template.get_task(
                oos_holdout_version=self.version,
                draw_id=draw_id,
            )
            forecast_task.add_upstream(scaling_task)
            forecast_task.add_downstream(forecast_join_task)
            self.workflow.add_task(forecast_task)

        # for measure in measures:
        #     task = postprocess_template.get_task(
        #         oos_holdout_version=self.version,
        #         measure=measure,
        #     )
        #     task.add_upstream(forecast_join_task)
        #     task.add_downstream(postprocess_join_task)
        #     self.workflow.add_task(task)
        #
        # for plot_type in plot_types:
        #     task = diagnostics_template.get_task(
        #         oos_holdout_version=self.version,
        #         plot_type=plot_type,
        #     )
        #     task.add_upstream(postprocess_join_task)
        #     self.workflow.add_task(task)

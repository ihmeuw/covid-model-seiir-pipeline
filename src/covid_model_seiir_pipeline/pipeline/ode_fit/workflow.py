import shutil

from covid_shared import workflow

import covid_model_seiir_pipeline
from covid_model_seiir_pipeline.pipeline.ode_fit.specification import ODE_FIT_JOBS


class ODEFitTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{ODE_FIT_JOBS.ode_fit}_draw_{{draw_id}}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{ODE_FIT_JOBS.ode_fit} "
            "--ode-fit-version {ode_fit_version} "
            "--draw-id {draw_id} "
            "-vv"
    )
    node_args = ['draw_id']
    task_args = ['ode_fit_version']


class HospitalCorrectionFactorTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{ODE_FIT_JOBS.synthesis_spline}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{ODE_FIT_JOBS.synthesis_spline} "
            "--ode-fit-version {ode_fit_version} "
            "--draw-id {draw_id} "
            "-vv"
    )
    node_args = []
    task_args = ['ode_fit_version']


class ODEFitWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-ode-fit-{version}'
    task_template_classes = {
        ODE_FIT_JOBS.ode_fit: ODEFitTaskTemplate,
    }

    def attach_tasks(self, n_draws: int):
        pass
from typing import List

from jobmon.client.swarm.executors.base import ExecutorParameters

from covid_model_seiir_pipeline.workflow_template import TaskTemplate, WorkflowTemplate

# TODO: Extract these into specification, maybe.  At least allow overrides
#    for the queue from the command line.
FORECAST_RUNTIME = 2000
FORECAST_MEMORY = '5G'
POSTPROCESS_MEMORY = '50G'
FORECAST_CORES = 1
FORECAST_SCALING_CORES = 50
FORECAST_QUEUE = 'd.q'


class BetaResidualScalingTaskTemplate(TaskTemplate):

    task_name_template = 'beta_residual_scaling_{scenario}'
    command_template = (
            "beta_residual_scaling " +
            "--forecast-version {forecast_version} " +
            "--scenario-name {scenario}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=FORECAST_RUNTIME,
        m_mem_free=FORECAST_MEMORY,
        num_cores=FORECAST_SCALING_CORES,
        queue=FORECAST_QUEUE
    )


class BetaForecastTaskTemplate(TaskTemplate):

    task_name_template = "forecast_{scenario}_{draw_id}"
    command_template = (
        "beta_forecast " +
        "--draw-id {draw_id} " +
        "--forecast-version {forecast_version} " +
        "--scenario-name {scenario}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=FORECAST_RUNTIME,
        m_mem_free=FORECAST_MEMORY,
        num_cores=FORECAST_CORES,
        queue=FORECAST_QUEUE
    )


class ConcatenateDrawsTaskTemplate(TaskTemplate):
    task_name_template = "seiir_concatenate_draws"
    command_template = (
            "concatenate " +
            "--forecast-version {forecast_version} "
            "--scenario-name {scenario}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=FORECAST_RUNTIME,
        m_mem_free=POSTPROCESS_MEMORY,
        num_cores=FORECAST_SCALING_CORES,
        queue=FORECAST_QUEUE
    )


class PostprocessingTaskTemplate(TaskTemplate):
    task_name_template = "seiir_post_processing"
    command_template = (
            "postprocess " +
            "--forecast-version {forecast_version} "
            "--scenario-name {scenario}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=FORECAST_RUNTIME,
        m_mem_free=POSTPROCESS_MEMORY,
        num_cores=FORECAST_SCALING_CORES,
        queue=FORECAST_QUEUE
    )


class ForecastWorkflow(WorkflowTemplate):

    workflow_name_template = 'seiir-forecast-{version}'
    task_templates = {
        'scaling': BetaResidualScalingTaskTemplate,
        'forecast': BetaForecastTaskTemplate,
        'concatenate': ConcatenateDrawsTaskTemplate,
        'postprocess': PostprocessingTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, scenarios: List[str]):
        scaling_template = self.task_templates['scaling']
        forecast_template = self.task_templates['forecast']
        concatenate_template = self.task_templates['concatenate']
        postprocessing_template = self.task_templates['postprocess']

        concatenate_tasks = {}
        for scenario in scenarios:
            concatenate_task = concatenate_template.get_task(
                forecast_version=self.version,
                scenario=scenario,
            )
            self.workflow.add_task(concatenate_task)
            concatenate_tasks['scenario'] = concatenate_task

        for scenario in scenarios:
            scaling_task = scaling_template.get_task(
                forecast_version=self.version,
                scenario=scenario
            )
            self.workflow.add_task(scaling_task)

            postprocessing_task = postprocessing_template.get_task(
                forecast_version=self.version
            )
            for concatenate_task in concatenate_tasks.values():
                postprocessing_task.add_upstream(concatenate_task)

            for draw in range(n_draws):
                forecast_task = forecast_template.get_task(
                    forecast_version=self.version,
                    draw_id=draw,
                    scenario=scenario
                )
                forecast_task.add_upstream(scaling_task)
                forecast_task.add_downstream(concatenate_tasks[scenario])
                self.workflow.add_task(forecast_task)

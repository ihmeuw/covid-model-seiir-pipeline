import itertools
from typing import List

from jobmon.client.swarm.executors.base import ExecutorParameters

from covid_model_seiir_pipeline.workflow_template import (
    TaskTemplate,
    WorkflowTemplate
)
from covid_model_seiir_pipeline.forecasting.specification import (
    FORECAST_QUEUE,
    FORECAST_RUNTIME,
    FORECAST_MEMORY,
    FORECAST_SCALING_CORES,
    FORECAST_CORES,
    POSTPROCESS_MEMORY
)
from covid_model_seiir_pipeline.forecasting.task.postprocessing import (
    MEASURES,
    MISCELLANEOUS,
    COVARIATES
)


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


class ResampleMapTaskTemplate(TaskTemplate):
    task_name_template = "seiir_resample_map"
    command_template = (
            "resample_map " +
            "--forecast-version {forecast_version} "
    )
    params = ExecutorParameters(
        max_runtime_seconds=FORECAST_RUNTIME,
        m_mem_free=POSTPROCESS_MEMORY,
        num_cores=FORECAST_SCALING_CORES,
        queue=FORECAST_QUEUE
    )


class PostprocessingTaskTemplate(TaskTemplate):
    task_name_template = "{measure}_{scenario}_post_processing"
    command_template = (
            "postprocess " +
            "--forecast-version {forecast_version} "
            "--scenario-name {scenario} "
            "--measure {measure}"
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
        'resample': ResampleMapTaskTemplate,
        'postprocess': PostprocessingTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, scenarios: List[str], covariates: List[str]):
        scaling_template = self.task_templates['scaling']
        forecast_template = self.task_templates['forecast']
        resample_template = self.task_templates['resample']
        postprocessing_template = self.task_templates['postprocess']

        resample_task = resample_template.get_task(
            forecast_version=self.version
        )
        self.workflow.add_task(resample_task)

        for scenario in scenarios:
            scaling_task = scaling_template.get_task(
                forecast_version=self.version,
                scenario=scenario
            )
            self.workflow.add_task(scaling_task)

            covariates = set(covariates).difference(['intercept'])
            unknown_covariates = covariates.difference(COVARIATES)
            if unknown_covariates:
                raise NotImplementedError(f'Unknown covariates {unknown_covariates}')

            measures = itertools.chain(MEASURES, MISCELLANEOUS, covariates)
            for measure in measures:
                postprocessing_task = postprocessing_template.get_task(
                    forecast_version=self.version,
                    scenario=scenario,
                    measure=measure,
                )
                postprocessing_task.add_upstream(resample_task)
                self.workflow.add_task(postprocessing_task)

            for draw in range(n_draws):
                forecast_task = forecast_template.get_task(
                    forecast_version=self.version,
                    draw_id=draw,
                    scenario=scenario
                )
                forecast_task.add_upstream(scaling_task)
                forecast_task.add_downstream(resample_task)
                self.workflow.add_task(forecast_task)

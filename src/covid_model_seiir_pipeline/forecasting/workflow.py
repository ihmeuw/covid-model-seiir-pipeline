import itertools
import shutil
from typing import Dict, List

from jobmon.client import BashTask

from covid_model_seiir_pipeline.workflow_tools.template import (
    TaskTemplate,
    WorkflowTemplate,
)
from covid_model_seiir_pipeline.forecasting.specification import (
    ScenarioSpecification,
    FORECAST_JOBS,
)
from covid_model_seiir_pipeline.forecasting.task.postprocessing import (
    MEASURES,
    MISCELLANEOUS,
    COVARIATES,
)


class BetaResidualScalingTaskTemplate(TaskTemplate):

    task_name_template = 'beta_residual_scaling_{scenario}'
    command_template = (
            f"{shutil.which('beta_residual_scaling')} " +
            "--forecast-version {forecast_version} " +
            "--scenario-name {scenario}"
    )


class BetaForecastTaskTemplate(TaskTemplate):

    task_name_template = "forecast_{scenario}_{draw_id}"
    command_template = (
        f"{shutil.which('beta_forecast')} " +
        "--draw-id {draw_id} " +
        "--forecast-version {forecast_version} " +
        "--scenario-name {scenario} " +
        "--extra-id {extra_id}"
    )


class ResampleMapTaskTemplate(TaskTemplate):
    task_name_template = "seiir_resample_map"
    command_template = (
            f"{shutil.which('resample_map')} " +
            "--forecast-version {forecast_version} "
    )


class PostprocessingTaskTemplate(TaskTemplate):

    task_name_template = "{measure}_{scenario}_post_processing"
    command_template = (
            f"{shutil.which('postprocess')} " +
            "--forecast-version {forecast_version} "
            "--scenario-name {scenario} "
            "--measure {measure}"
    )


class JoinSentinelTaskTemplate(TaskTemplate):
    """Dummy task to simplify DAG."""
    sentinel_id = 1
    task_name_template = "join_sentinel_{sentinel_id}"
    command_template = (
        "echo {sentinel_id}"
    )

    def get_task(self, *args, **kwargs):
        task = super().get_task(*args, sentinel_id=self.sentinel_id, **kwargs)
        self.sentinel_id += 1
        return task


class ForecastWorkflow(WorkflowTemplate):

    workflow_name_template = 'seiir-forecast-{version}'
    task_template_classes = {
        FORECAST_JOBS.scaling: BetaResidualScalingTaskTemplate,
        FORECAST_JOBS.forecast: BetaForecastTaskTemplate,
        FORECAST_JOBS.resample: ResampleMapTaskTemplate,
        FORECAST_JOBS.postprocess: PostprocessingTaskTemplate,
        FORECAST_JOBS.sentinel: JoinSentinelTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, scenarios: Dict[str, ScenarioSpecification], covariates: List[str]):
        scaling_template = self.task_templates[FORECAST_JOBS.scaling]
        resample_template = self.task_templates[FORECAST_JOBS.resample]

        # The draw resampling map is produced for one reference scenario
        # after the forecasts and then used to postprocess all measures for
        # all scenarios.
        resample_task = resample_template.get_task(
            forecast_version=self.version
        )
        self.workflow.add_task(resample_task)

        for scenario_name, scenario_spec in scenarios.items():
            # Computing the beta scaling parameters is the first step for each
            # scenario forecast.
            scaling_task = scaling_template.get_task(
                forecast_version=self.version,
                scenario=scenario_name
            )

            self.workflow.add_task(scaling_task)

            forecast_done_task = self._attach_forecast_tasks(scenario_name, n_draws, 0, scaling_task)
            self._attach_postprocessing_tasks(scenario_name, covariates, forecast_done_task, resample_task)
            resample_task.add_upstream(forecast_done_task)

    def _attach_forecast_tasks(self, scenario_name: str, n_draws: int, extra_id: int,
                               *upstream_tasks: BashTask) -> BashTask:
        forecast_template = self.task_templates[FORECAST_JOBS.forecast]
        sentinel_template = self.task_templates[FORECAST_JOBS.sentinel]

        sentinel_task = sentinel_template.get_task()
        self.workflow.add_task(sentinel_task)

        for draw in range(n_draws):
            forecast_task = forecast_template.get_task(
                forecast_version=self.version,
                draw_id=draw,
                scenario=scenario_name,
                extra_id=extra_id,
            )
            for upstream_task in upstream_tasks:
                forecast_task.add_upstream(upstream_task)
            forecast_task.add_downstream(sentinel_task)
            self.workflow.add_task(forecast_task)
        return sentinel_task

    def _attach_postprocessing_tasks(self, scenario_name: str, covariates: List[str],
                                     *upstream_tasks: BashTask) -> None:
        postprocessing_template = self.task_templates[FORECAST_JOBS.postprocess]

        covariates = set(covariates).difference(['intercept'])
        unknown_covariates = covariates.difference(COVARIATES)
        if unknown_covariates:
            raise NotImplementedError(f'Unknown covariates {unknown_covariates}')

        measures = itertools.chain(MEASURES, MISCELLANEOUS, covariates)
        for measure in measures:
            postprocessing_task = postprocessing_template.get_task(
                forecast_version=self.version,
                scenario=scenario_name,
                measure=measure,
            )
            for upstream_task in upstream_tasks:
                postprocessing_task.add_upstream(upstream_task)
            self.workflow.add_task(postprocessing_task)

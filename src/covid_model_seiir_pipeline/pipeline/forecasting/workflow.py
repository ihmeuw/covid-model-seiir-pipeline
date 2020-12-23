import shutil
from typing import Dict

from covid_model_seiir_pipeline.lib import workflow
from covid_model_seiir_pipeline.lib.ihme_deps import BashTask
from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
    ScenarioSpecification,
    FORECAST_JOBS,
)


class BetaResidualScalingTaskTemplate(workflow.TaskTemplate):

    task_name_template = 'beta_residual_scaling_{scenario}'
    command_template = (
            f"{shutil.which('beta_residual_scaling')} " +
            "--forecast-version {forecast_version} " +
            "--scenario-name {scenario}"
    )


class BetaForecastTaskTemplate(workflow.TaskTemplate):

    task_name_template = "forecast_{scenario}_{draw_id}"
    command_template = (
        f"{shutil.which('beta_forecast')} " +
        "--draw-id {draw_id} " +
        "--forecast-version {forecast_version} " +
        "--scenario-name {scenario} " +
        "--extra-id {extra_id}"
    )


class ForecastWorkflow(workflow.WorkflowTemplate):

    workflow_name_template = 'seiir-forecast-{version}'
    task_template_classes = {
        FORECAST_JOBS.scaling: BetaResidualScalingTaskTemplate,
        FORECAST_JOBS.forecast: BetaForecastTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, scenarios: Dict[str, ScenarioSpecification]):
        scaling_template = self.task_templates[FORECAST_JOBS.scaling]

        for scenario_name, scenario_spec in scenarios.items():
            # Computing the beta scaling parameters is the first step for each
            # scenario forecast.
            scaling_task = scaling_template.get_task(
                forecast_version=self.version,
                scenario=scenario_name
            )
            self.workflow.add_task(scaling_task)
            self._attach_forecast_tasks(scenario_name, n_draws, 0, scaling_task)

    def _attach_forecast_tasks(self, scenario_name: str, n_draws: int, extra_id: int,
                               *upstream_tasks: BashTask) -> None:
        forecast_template = self.task_templates[FORECAST_JOBS.forecast]

        for draw in range(n_draws):
            forecast_task = forecast_template.get_task(
                forecast_version=self.version,
                draw_id=draw,
                scenario=scenario_name,
                extra_id=extra_id,
            )
            for upstream_task in upstream_tasks:
                forecast_task.add_upstream(upstream_task)
            self.workflow.add_task(forecast_task)

from jobmon.client.swarm.executors.base import ExecutorParameters

from covid_model_seiir_pipeline.workflow_template import TaskTemplate, WorkflowTemplate


class OOSRegressionTaskTemplate(TaskTemplate):
    task_name_template = "oos_regression_wrapper"
    command_template = (
            "oos_regression " +
            "--regression-specification-path {regression_specification_path}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=50000,
        m_mem_free='2G',
        num_cores=1,
        queue='d.q'
    )


class OOSForecastTaskTemplate(TaskTemplate):
    task_name_template = "oos_forecast_wrapper"
    command_template = (
            "oos_forecast " +
            "--forecast-specification-path {forecast_specification_path}"
    )
    params = ExecutorParameters(
        max_runtime_seconds=50000,
        m_mem_free='2G',
        num_cores=1,
        queue='d.q'
    )


class PredictiveValidityWorkflow(WorkflowTemplate):

    workflow_name_template = 'predictive-validity'
    task_templates = {'regression': OOSRegressionTaskTemplate,
                      'forecast': OOSForecastTaskTemplate}

    def attach_tasks(self, regression_specification_path, forecast_specification_paths):
        regression_template = self.task_templates['regression']
        forecast_template = self.task_templates['forecast']
        regression_task = regression_template.get_task(
            regression_specification_path=regression_specification_path
        )
        self.workflow.add_task(regression_task)
        for forecast_specification_path in forecast_specification_paths:
            forecast_task = forecast_template.get_task(
                forecast_specification_path=forecast_specification_path
            )
            self.workflow.add_task(forecast_task)
            forecast_task.add_upstream(regression_task)

from jobmon.client.swarm.workflow.bash_task import BashTask
from jobmon.client.swarm.executors.base import ExecutorParameters


ExecParams = ExecutorParameters(
    max_runtime_seconds=1000,
    j_resource=False,
    m_mem_free='20G',
    num_cores=3,
    queue='d.q'
)


class RegressionTask(BashTask):
    def __init__(self, draw_id, regression_version, **kwargs):

        self.draw_id = draw_id

        command = (
            "beta_regression " +
            f"--draw-id {self.draw_id} " +
            f"--regression-version {regression_version} "
        )

        super().__init__(
            command=command,
            name=f'seiir_regression_fit_{draw_id}',
            executor_parameters=ExecParams,
            max_attempts=1,
            **kwargs
        )


class ForecastTask(BashTask):
    def __init__(self, location_id, regression_version, forecast_version, **kwargs):

        self.location_id = location_id

        command = (
            "beta_forecast " +
            f"--location-id {self.location_id} " +
            f"--regression-version {regression_version} " +
            f"--forecast-version {forecast_version} "
        )

        super().__init__(
            command=command,
            name=f'seiir_forecast_location_{location_id}',
            executor_parameters=ExecParams,
            max_attempts=1,
            **kwargs
        )
